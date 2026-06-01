use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use claudius::{Anthropic, MessageCreateParams, Model};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict, PyFloat, PyInt, PyList, PyModule, PyString, PyTuple};

use crate::{
    ApplyError, Conflict, Field, InferenceConfig, Manager, OnConflict, Policy, PolicyType, Report,
    Usage, DEFAULT_MODEL,
};

#[pyclass(name = "Client")]
struct PyClient {
    inner: Arc<Anthropic>,
}

#[pymethods]
impl PyClient {
    #[new]
    #[pyo3(signature = (api_key=None))]
    fn new(api_key: Option<String>) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(Anthropic::new(api_key).map_err(runtime_error)?),
        })
    }

    fn __repr__(&self) -> &'static str {
        "Client()"
    }
}

#[pyclass(name = "Field")]
#[derive(Clone)]
struct PyField {
    inner: Field,
}

#[pymethods]
impl PyField {
    #[staticmethod]
    #[pyo3(name = "bool", signature = (name, default=None, on_conflict="default"))]
    fn bool_field(
        name: String,
        default: Option<bool>,
        on_conflict: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Field::Bool {
                name,
                default,
                on_conflict: parse_on_conflict(on_conflict)?,
            },
        })
    }

    #[staticmethod]
    #[pyo3(signature = (name, default=None, on_conflict="default"))]
    fn number(name: String, default: Option<f64>, on_conflict: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Field::Number {
                name,
                default: default.map(crate::t64),
                on_conflict: parse_on_conflict(on_conflict)?,
            },
        })
    }

    #[staticmethod]
    #[pyo3(signature = (name, default=None, on_conflict="default"))]
    fn string(name: String, default: Option<String>, on_conflict: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Field::String {
                name,
                default,
                on_conflict: parse_on_conflict(on_conflict)?,
            },
        })
    }

    #[staticmethod]
    #[pyo3(name = "enum", signature = (name, values, default=None, on_conflict="default"))]
    fn string_enum(
        name: String,
        values: Vec<String>,
        default: Option<String>,
        on_conflict: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Field::StringEnum {
                name,
                values,
                default,
                on_conflict: parse_on_conflict(on_conflict)?,
            },
        })
    }

    #[staticmethod]
    fn string_array(name: String) -> Self {
        Self {
            inner: Field::StringArray { name },
        }
    }

    #[staticmethod]
    fn from_dict(mapping: &Bound<'_, PyAny>) -> PyResult<Self> {
        let value = py_to_json(mapping)?;
        Ok(Self {
            inner: field_from_value(&value)?,
        })
    }

    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }

    #[getter]
    fn kind(&self) -> &'static str {
        field_kind(&self.inner)
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &field_to_value(&self.inner))
    }

    fn __repr__(&self) -> String {
        format!("Field({})", field_to_value(&self.inner))
    }
}

#[pyclass(name = "PolicyType")]
#[derive(Clone)]
struct PyPolicyType {
    inner: PolicyType,
}

#[pymethods]
impl PyPolicyType {
    #[new]
    fn new(name: String, fields: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut parsed_fields = Vec::new();
        for item in fields.try_iter()? {
            parsed_fields.push(field_from_py(&item?)?);
        }
        Ok(Self {
            inner: PolicyType {
                name,
                fields: parsed_fields,
            },
        })
    }

    #[staticmethod]
    fn parse(source: &str) -> PyResult<Self> {
        Ok(Self {
            inner: PolicyType::parse(source).map_err(value_error)?,
        })
    }

    #[staticmethod]
    fn from_dict(mapping: &Bound<'_, PyAny>) -> PyResult<Self> {
        let value = py_to_json(mapping)?;
        Ok(Self {
            inner: policy_type_from_value(&value)?,
        })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn fields(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let value = serde_json::Value::Array(
            self.inner
                .fields
                .iter()
                .map(field_to_value)
                .collect::<Vec<_>>(),
        );
        json_to_py(py, &value)
    }

    fn default_value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &self.inner.default_value())
    }

    #[pyo3(signature = (client, injection, model=None))]
    fn with_semantic_injection(
        &self,
        py: Python<'_>,
        client: PyRef<'_, PyClient>,
        injection: &str,
        model: Option<&str>,
    ) -> PyResult<PyPolicy> {
        let model = parse_model(model)?;
        let client = Arc::clone(&client.inner);
        let policy_type = self.inner.clone();
        let injection = injection.to_string();
        let runtime = build_runtime()?;
        let policy = py
            .allow_threads(|| {
                runtime.block_on(policy_type.with_semantic_injection(
                    client.as_ref(),
                    &injection,
                    model,
                )).map_err(|error| error.to_string())
            })
            .map_err(runtime_error)?;
        Ok(PyPolicy { inner: policy })
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &policy_type_to_value(&self.inner))
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("PolicyType.parse({:?})", self.inner.to_string())
    }
}

#[pyclass(name = "Policy")]
#[derive(Clone)]
struct PyPolicy {
    inner: Policy,
}

#[pymethods]
impl PyPolicy {
    #[new]
    fn new(
        policy_type: PyRef<'_, PyPolicyType>,
        prompt: String,
        action: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Policy {
                r#type: policy_type.inner.clone(),
                prompt,
                action: py_to_json(action)?,
            },
        })
    }

    #[staticmethod]
    fn from_dict(mapping: &Bound<'_, PyAny>) -> PyResult<Self> {
        let value = py_to_json(mapping)?;
        Ok(Self {
            inner: policy_from_value(&value)?,
        })
    }

    #[getter]
    fn policy_type(&self) -> PyPolicyType {
        PyPolicyType {
            inner: self.inner.r#type.clone(),
        }
    }

    #[getter]
    fn prompt(&self) -> &str {
        &self.inner.prompt
    }

    #[getter]
    fn action(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &self.inner.action)
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &policy_to_value(&self.inner))
    }

    fn __repr__(&self) -> String {
        format!("Policy(prompt={:?}, action={})", self.inner.prompt, self.inner.action)
    }
}

#[pyclass(name = "Manager")]
struct PyManager {
    inner: Manager,
    policy_count: usize,
}

#[pymethods]
impl PyManager {
    #[new]
    #[pyo3(signature = (policies=None, inference_config="json_schema"))]
    fn new(policies: Option<&Bound<'_, PyAny>>, inference_config: &str) -> PyResult<Self> {
        let mut manager = Manager::default();
        manager.set_inference_config(parse_inference_config(inference_config)?);
        let mut policy_count = 0;
        if let Some(policies) = policies {
            for item in policies.try_iter()? {
                manager
                    .try_add(policy_from_py(&item?)?)
                    .map_err(value_error)?;
                policy_count += 1;
            }
        }
        Ok(Self {
            inner: manager,
            policy_count,
        })
    }

    fn add(&mut self, policy: PyRef<'_, PyPolicy>) -> PyResult<()> {
        self.inner
            .try_add(policy.inner.clone())
            .map_err(value_error)?;
        self.policy_count += 1;
        Ok(())
    }

    #[getter]
    fn inference_config(&self) -> &'static str {
        inference_config_to_str(self.inner.inference_config())
    }

    #[setter]
    fn set_inference_config(&mut self, inference_config: &str) -> PyResult<()> {
        self.inner
            .set_inference_config(parse_inference_config(inference_config)?);
        Ok(())
    }

    #[pyo3(signature = (
        client,
        text,
        *,
        model=None,
        max_tokens=4096,
        temperature=None,
        track_usage=false,
        inference_config=None
    ))]
    fn apply(
        &mut self,
        py: Python<'_>,
        client: PyRef<'_, PyClient>,
        text: &str,
        model: Option<&str>,
        max_tokens: u32,
        temperature: Option<f32>,
        track_usage: bool,
        inference_config: Option<&str>,
    ) -> PyResult<PyReport> {
        let inference_config = match inference_config {
            Some(inference_config) => parse_inference_config(inference_config)?,
            None => self.inner.inference_config(),
        };
        apply_with_client(
            &mut self.inner,
            py,
            Arc::clone(&client.inner),
            text,
            model,
            max_tokens,
            temperature,
            track_usage,
            inference_config,
        )
    }

    fn __len__(&self) -> usize {
        self.policy_count
    }

    fn __repr__(&self) -> String {
        format!(
            "Manager(policies={}, inference_config={:?})",
            self.policy_count,
            inference_config_to_str(self.inner.inference_config())
        )
    }
}

#[pyclass(name = "Report")]
struct PyReport {
    inner: Report,
    usage: Option<Usage>,
}

#[pymethods]
impl PyReport {
    #[getter]
    fn value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &self.inner.value())
    }

    #[getter]
    fn ir(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(
            py,
            self.inner
                .ir
                .as_ref()
                .unwrap_or(&serde_json::Value::Null),
        )
    }

    #[getter]
    fn default(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(
            py,
            self.inner
                .default
                .as_ref()
                .unwrap_or(&serde_json::Value::Null),
        )
    }

    #[getter]
    fn rules_matched(&self) -> Vec<usize> {
        self.inner.rules_matched.clone()
    }

    #[getter]
    fn errors(&self) -> Vec<String> {
        report_errors(&self.inner)
    }

    #[getter]
    fn conflicts(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let value = serde_json::Value::Array(report_conflicts(&self.inner));
        json_to_py(py, &value)
    }

    #[getter]
    fn field_stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &to_json_value(self.inner.all_field_stats())?)
    }

    #[getter]
    fn usage(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(
            py,
            &self
                .usage
                .as_ref()
                .map(usage_to_value)
                .unwrap_or(serde_json::Value::Null),
        )
    }

    fn has_errors(&self) -> bool {
        self.inner.has_errors()
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &report_to_value(&self.inner, self.usage.as_ref())?)
    }

    fn __repr__(&self) -> String {
        format!("Report(value={})", self.inner.value())
    }
}

#[pyfunction(name = "bool_field")]
#[pyo3(signature = (name, default=None, on_conflict="default"))]
fn py_bool_field(name: String, default: Option<bool>, on_conflict: &str) -> PyResult<PyField> {
    PyField::bool_field(name, default, on_conflict)
}

#[pyfunction(name = "number_field")]
#[pyo3(signature = (name, default=None, on_conflict="default"))]
fn py_number_field(name: String, default: Option<f64>, on_conflict: &str) -> PyResult<PyField> {
    PyField::number(name, default, on_conflict)
}

#[pyfunction(name = "string_field")]
#[pyo3(signature = (name, default=None, on_conflict="default"))]
fn py_string_field(name: String, default: Option<String>, on_conflict: &str) -> PyResult<PyField> {
    PyField::string(name, default, on_conflict)
}

#[pyfunction(name = "enum_field")]
#[pyo3(signature = (name, values, default=None, on_conflict="default"))]
fn py_enum_field(
    name: String,
    values: Vec<String>,
    default: Option<String>,
    on_conflict: &str,
) -> PyResult<PyField> {
    PyField::string_enum(name, values, default, on_conflict)
}

#[pyfunction(name = "string_array_field")]
fn py_string_array_field(name: String) -> PyField {
    PyField::string_array(name)
}

#[pyfunction]
fn parse_policy_type(source: &str) -> PyResult<PyPolicyType> {
    PyPolicyType::parse(source)
}

#[pyfunction]
#[pyo3(signature = (
    policies,
    text,
    *,
    client=None,
    model=None,
    max_tokens=4096,
    temperature=None,
    track_usage=false,
    inference_config="json_schema"
))]
fn apply(
    py: Python<'_>,
    policies: &Bound<'_, PyAny>,
    text: &str,
    client: Option<PyRef<'_, PyClient>>,
    model: Option<&str>,
    max_tokens: u32,
    temperature: Option<f32>,
    track_usage: bool,
    inference_config: &str,
) -> PyResult<PyReport> {
    let mut manager = PyManager::new(Some(policies), inference_config)?;
    let inference_config = manager.inner.inference_config();
    let client = match client {
        Some(client) => Arc::clone(&client.inner),
        None => PyClient::new(None)?.inner,
    };
    apply_with_client(
        &mut manager.inner,
        py,
        client,
        text,
        model,
        max_tokens,
        temperature,
        track_usage,
        inference_config,
    )
}

#[pymodule]
fn policyai(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyClient>()?;
    m.add_class::<PyField>()?;
    m.add_class::<PyPolicyType>()?;
    m.add_class::<PyPolicy>()?;
    m.add_class::<PyManager>()?;
    m.add_class::<PyReport>()?;
    m.add_function(wrap_pyfunction!(py_bool_field, m)?)?;
    m.add_function(wrap_pyfunction!(py_number_field, m)?)?;
    m.add_function(wrap_pyfunction!(py_string_field, m)?)?;
    m.add_function(wrap_pyfunction!(py_enum_field, m)?)?;
    m.add_function(wrap_pyfunction!(py_string_array_field, m)?)?;
    m.add_function(wrap_pyfunction!(parse_policy_type, m)?)?;
    m.add_function(wrap_pyfunction!(apply, m)?)?;
    m.add("DEFAULT_MODEL", DEFAULT_MODEL.to_string())?;
    m.add("DEFAULT", "default")?;
    m.add("AGREEMENT", "agreement")?;
    m.add("LARGEST_VALUE", "largest")?;
    m.add("JSON_SCHEMA", "json_schema")?;
    m.add("TOOL_USE", "tool_use")?;
    m.add("STRICT_TOOL_USE", "strict_tool_use")?;
    Ok(())
}

fn parse_on_conflict(value: &str) -> PyResult<OnConflict> {
    match normalized(value).as_str() {
        "default" => Ok(OnConflict::Default),
        "agreement" | "agree" | "must_agree" => Ok(OnConflict::Agreement),
        "largest" | "largest_value" | "highest" | "highest_wins" | "sticky" | "last_wins" => {
            Ok(OnConflict::LargestValue)
        }
        _ => Err(PyValueError::new_err(format!(
            "unknown conflict strategy {value:?}; expected 'default', 'agreement', or 'largest'"
        ))),
    }
}

fn on_conflict_to_str(on_conflict: OnConflict) -> &'static str {
    match on_conflict {
        OnConflict::Default => "default",
        OnConflict::Agreement => "agreement",
        OnConflict::LargestValue => "largest",
    }
}

fn parse_inference_config(value: &str) -> PyResult<InferenceConfig> {
    match normalized(value).as_str() {
        "json_schema" | "output_format" | "output_format_json_schema" => {
            Ok(InferenceConfig::OutputFormatJsonSchema)
        }
        "tool" | "tool_use" => Ok(InferenceConfig::ToolUse),
        "strict" | "strict_tool" | "strict_tool_use" => Ok(InferenceConfig::StrictToolUse),
        _ => Err(PyValueError::new_err(format!(
            "unknown inference_config {value:?}; expected 'json_schema', 'tool_use', or 'strict_tool_use'"
        ))),
    }
}

fn inference_config_to_str(inference_config: InferenceConfig) -> &'static str {
    match inference_config {
        InferenceConfig::ToolUse => "tool_use",
        InferenceConfig::StrictToolUse => "strict_tool_use",
        InferenceConfig::OutputFormatJsonSchema => "json_schema",
    }
}

fn parse_model(model: Option<&str>) -> PyResult<Model> {
    model
        .map(Model::from_str)
        .transpose()
        .map_err(|_| PyValueError::new_err("invalid model"))?
        .map(Ok)
        .unwrap_or(Ok(DEFAULT_MODEL))
}

fn normalized(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .replace(['-', ' '], "_")
}

fn field_from_py(item: &Bound<'_, PyAny>) -> PyResult<Field> {
    if let Ok(field) = item.extract::<PyRef<'_, PyField>>() {
        return Ok(field.inner.clone());
    }
    field_from_value(&py_to_json(item)?)
}

fn policy_from_py(item: &Bound<'_, PyAny>) -> PyResult<Policy> {
    if let Ok(policy) = item.extract::<PyRef<'_, PyPolicy>>() {
        return Ok(policy.inner.clone());
    }
    policy_from_value(&py_to_json(item)?)
}

#[derive(Clone, Debug, PartialEq)]
enum FieldWire {
    Bool {
        name: String,
        default: Option<bool>,
        on_conflict: OnConflict,
    },
    Number {
        name: String,
        default: Option<f64>,
        on_conflict: OnConflict,
    },
    String {
        name: String,
        default: Option<String>,
        on_conflict: OnConflict,
    },
    StringEnum {
        name: String,
        values: Vec<String>,
        default: Option<String>,
        on_conflict: OnConflict,
    },
    StringArray {
        name: String,
    },
}

impl FieldWire {
    fn from_value(value: &serde_json::Value) -> PyResult<Self> {
        let object = value
            .as_object()
            .ok_or_else(|| PyTypeError::new_err("field must be a dict or Field"))?;
        let kind = object
            .get("type")
            .or_else(|| object.get("kind"))
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| PyValueError::new_err("field dict must include 'type'"))?;
        let name = object
            .get("name")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| PyValueError::new_err("field dict must include string 'name'"))?
            .to_string();
        let on_conflict = object
            .get("on_conflict")
            .or_else(|| object.get("conflict"))
            .and_then(serde_json::Value::as_str)
            .map(parse_on_conflict)
            .transpose()?
            .unwrap_or_default();

        match normalized(kind).as_str() {
            "bool" | "boolean" => Ok(Self::Bool {
                name,
                default: optional_bool(object.get("default"))?,
                on_conflict,
            }),
            "number" | "float" | "int" | "integer" => Ok(Self::Number {
                name,
                default: optional_number(object.get("default"))?,
                on_conflict,
            }),
            "string" | "str" => Ok(Self::String {
                name,
                default: optional_string(object.get("default"))?,
                on_conflict,
            }),
            "enum" | "string_enum" => Ok(Self::StringEnum {
                name,
                values: required_string_array(object.get("values"), "enum field requires 'values'")?,
                default: optional_string(object.get("default"))?,
                on_conflict,
            }),
            "array" | "string_array" | "list" => Ok(Self::StringArray { name }),
            _ => Err(PyValueError::new_err(format!("unknown field type {kind:?}"))),
        }
    }

    fn into_field(self) -> Field {
        match self {
            Self::Bool {
                name,
                default,
                on_conflict,
            } => Field::Bool {
                name,
                default,
                on_conflict,
            },
            Self::Number {
                name,
                default,
                on_conflict,
            } => Field::Number {
                name,
                default: default.map(crate::t64),
                on_conflict,
            },
            Self::String {
                name,
                default,
                on_conflict,
            } => Field::String {
                name,
                default,
                on_conflict,
            },
            Self::StringEnum {
                name,
                values,
                default,
                on_conflict,
            } => Field::StringEnum {
                name,
                values,
                default,
                on_conflict,
            },
            Self::StringArray { name } => Field::StringArray { name },
        }
    }

    fn to_value(&self) -> serde_json::Value {
        match self {
            Self::Bool {
                name,
                default,
                on_conflict,
            } => serde_json::json!({
                "type": "bool",
                "name": name,
                "default": default,
                "on_conflict": on_conflict_to_str(*on_conflict),
            }),
            Self::Number {
                name,
                default,
                on_conflict,
            } => serde_json::json!({
                "type": "number",
                "name": name,
                "default": default,
                "on_conflict": on_conflict_to_str(*on_conflict),
            }),
            Self::String {
                name,
                default,
                on_conflict,
            } => serde_json::json!({
                "type": "string",
                "name": name,
                "default": default,
                "on_conflict": on_conflict_to_str(*on_conflict),
            }),
            Self::StringEnum {
                name,
                values,
                default,
                on_conflict,
            } => serde_json::json!({
                "type": "enum",
                "name": name,
                "values": values,
                "default": default,
                "on_conflict": on_conflict_to_str(*on_conflict),
            }),
            Self::StringArray { name } => serde_json::json!({
                "type": "string_array",
                "name": name,
            }),
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Self::Bool { .. } => "bool",
            Self::Number { .. } => "number",
            Self::String { .. } => "string",
            Self::StringEnum { .. } => "enum",
            Self::StringArray { .. } => "string_array",
        }
    }
}

impl From<&Field> for FieldWire {
    fn from(field: &Field) -> Self {
        match field {
            Field::Bool {
                name,
                default,
                on_conflict,
            } => Self::Bool {
                name: name.clone(),
                default: *default,
                on_conflict: *on_conflict,
            },
            Field::Number {
                name,
                default,
                on_conflict,
            } => Self::Number {
                name: name.clone(),
                default: default.map(|value| value.0),
                on_conflict: *on_conflict,
            },
            Field::String {
                name,
                default,
                on_conflict,
            } => Self::String {
                name: name.clone(),
                default: default.clone(),
                on_conflict: *on_conflict,
            },
            Field::StringEnum {
                name,
                values,
                default,
                on_conflict,
            } => Self::StringEnum {
                name: name.clone(),
                values: values.clone(),
                default: default.clone(),
                on_conflict: *on_conflict,
            },
            Field::StringArray { name } => Self::StringArray { name: name.clone() },
        }
    }
}

fn field_from_value(value: &serde_json::Value) -> PyResult<Field> {
    Ok(FieldWire::from_value(value)?.into_field())
}

fn field_to_value(field: &Field) -> serde_json::Value {
    FieldWire::from(field).to_value()
}

fn field_kind(field: &Field) -> &'static str {
    FieldWire::from(field).kind()
}

fn policy_type_from_value(value: &serde_json::Value) -> PyResult<PolicyType> {
    if let Some(source) = value.as_str() {
        return PolicyType::parse(source).map_err(value_error);
    }
    let object = value
        .as_object()
        .ok_or_else(|| PyTypeError::new_err("policy type must be a dict, string, or PolicyType"))?;
    let name = object
        .get("name")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| PyValueError::new_err("policy type dict must include string 'name'"))?
        .to_string();
    let fields = object
        .get("fields")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| PyValueError::new_err("policy type dict must include list 'fields'"))?
        .iter()
        .map(field_from_value)
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PolicyType { name, fields })
}

fn policy_type_to_value(policy_type: &PolicyType) -> serde_json::Value {
    serde_json::json!({
        "name": policy_type.name,
        "fields": policy_type.fields.iter().map(field_to_value).collect::<Vec<_>>(),
    })
}

fn policy_from_value(value: &serde_json::Value) -> PyResult<Policy> {
    let object = value
        .as_object()
        .ok_or_else(|| PyTypeError::new_err("policy must be a dict or Policy"))?;
    let policy_type = object
        .get("policy_type")
        .or_else(|| object.get("type"))
        .ok_or_else(|| PyValueError::new_err("policy dict must include 'policy_type'"))?;
    let prompt = object
        .get("prompt")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| PyValueError::new_err("policy dict must include string 'prompt'"))?
        .to_string();
    let action = object
        .get("action")
        .ok_or_else(|| PyValueError::new_err("policy dict must include 'action'"))?
        .clone();
    Ok(Policy {
        r#type: policy_type_from_value(policy_type)?,
        prompt,
        action,
    })
}

fn policy_to_value(policy: &Policy) -> serde_json::Value {
    serde_json::json!({
        "policy_type": policy_type_to_value(&policy.r#type),
        "prompt": policy.prompt,
        "action": policy.action,
    })
}

fn optional_bool(value: Option<&serde_json::Value>) -> PyResult<Option<bool>> {
    match value {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(PyTypeError::new_err("default must be bool or None")),
    }
}

fn optional_number(value: Option<&serde_json::Value>) -> PyResult<Option<f64>> {
    match value {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::Number(value)) => value
            .as_f64()
            .ok_or_else(|| PyTypeError::new_err("default must fit in a Python float"))
            .map(Some),
        Some(_) => Err(PyTypeError::new_err("default must be number or None")),
    }
}

fn optional_string(value: Option<&serde_json::Value>) -> PyResult<Option<String>> {
    match value {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(PyTypeError::new_err("default must be str or None")),
    }
}

fn required_string_array(
    value: Option<&serde_json::Value>,
    missing_message: &'static str,
) -> PyResult<Vec<String>> {
    value
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| PyValueError::new_err(missing_message))?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| PyTypeError::new_err("values must be a list of strings"))
        })
        .collect()
}

fn conflict_to_value(conflict: &Conflict) -> serde_json::Value {
    match conflict {
        Conflict::BoolConflict { field, val1, val2 } => serde_json::json!({
            "type": "bool",
            "field": field,
            "values": [val1, val2],
        }),
        Conflict::NumberConflict { field, val1, val2 } => serde_json::json!({
            "type": "number",
            "field": field,
            "values": [val1, val2],
        }),
        Conflict::StringConflict { field, val1, val2 } => serde_json::json!({
            "type": "string",
            "field": field,
            "values": [val1, val2],
        }),
        Conflict::Disagree {
            name,
            value1,
            value2,
        } => serde_json::json!({
            "type": "disagree",
            "field": name,
            "values": [value1, value2],
        }),
    }
}

fn report_errors(report: &Report) -> Vec<String> {
    report.errors().iter().map(ToString::to_string).collect()
}

fn report_conflicts(report: &Report) -> Vec<serde_json::Value> {
    report.conflicts().iter().map(conflict_to_value).collect()
}

fn report_to_value(report: &Report, usage: Option<&Usage>) -> PyResult<serde_json::Value> {
    Ok(serde_json::json!({
        "value": report.value(),
        "ir": report.ir.clone(),
        "default": report.default.clone(),
        "rules_matched": report.rules_matched.clone(),
        "errors": report_errors(report),
        "conflicts": report_conflicts(report),
        "field_stats": to_json_value(report.all_field_stats())?,
        "usage": usage.map(usage_to_value),
    }))
}

fn usage_to_value(usage: &Usage) -> serde_json::Value {
    let claudius_usage = usage.claudius_usage.map(|usage| {
        serde_json::json!({
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_creation_input_tokens": usage.cache_creation_input_tokens,
            "cache_read_input_tokens": usage.cache_read_input_tokens,
            "server_tool_use": to_json_value(&usage.server_tool_use).ok(),
        })
    });
    serde_json::json!({
        "iterations": usage.iterations,
        "wall_clock_seconds": duration_seconds(usage.wall_clock_time),
        "claudius": claudius_usage,
    })
}

fn duration_seconds(duration: Duration) -> f64 {
    duration.as_secs() as f64 + f64::from(duration.subsec_nanos()) / 1_000_000_000.0
}

fn build_runtime() -> PyResult<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(runtime_error)
}

#[allow(clippy::too_many_arguments)]
fn apply_with_client(
    manager: &mut Manager,
    py: Python<'_>,
    client: Arc<Anthropic>,
    text: &str,
    model: Option<&str>,
    max_tokens: u32,
    temperature: Option<f32>,
    track_usage: bool,
    inference_config: InferenceConfig,
) -> PyResult<PyReport> {
    let mut template = MessageCreateParams::default();
    template.max_tokens = max_tokens;
    template.model = parse_model(model)?;
    if let Some(temperature) = temperature {
        if !(0.0..=1.0).contains(&temperature) || !temperature.is_finite() {
            return Err(PyValueError::new_err(
                "temperature must be a finite number between 0.0 and 1.0",
            ));
        }
        template.temperature = Some(temperature);
    }

    let runtime = build_runtime()?;
    let text = text.to_string();
    let mut usage = track_usage.then(Usage::new);
    let report = py
        .allow_threads(|| {
            runtime.block_on(manager.apply_with_inference_config(
                client.as_ref(),
                template,
                &text,
                usage.as_mut(),
                inference_config,
            )).map_err(apply_failure)
        })
        .map_err(failure_to_pyerr)?;
    Ok(PyReport {
        inner: report,
        usage,
    })
}

fn py_to_json(value: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if value.is_none() {
        return Ok(serde_json::Value::Null);
    }
    if value.is_instance_of::<PyBool>() {
        return Ok(serde_json::Value::Bool(value.extract()?));
    }
    if value.is_instance_of::<PyInt>() {
        if let Ok(integer) = value.extract::<i64>() {
            return Ok(serde_json::Value::Number(integer.into()));
        }
        if let Ok(integer) = value.extract::<u64>() {
            return Ok(serde_json::Value::Number(integer.into()));
        }
        return Err(PyValueError::new_err("Python int does not fit in a JSON number"));
    }
    if value.is_instance_of::<PyFloat>() {
        let number = value.extract::<f64>()?;
        return serde_json::Number::from_f64(number)
            .map(serde_json::Value::Number)
            .ok_or_else(|| PyValueError::new_err("JSON numbers must be finite"));
    }
    if value.is_instance_of::<PyString>() {
        return Ok(serde_json::Value::String(value.extract()?));
    }
    if let Ok(sequence) = value.downcast::<PyList>() {
        return sequence
            .iter()
            .map(|item| py_to_json(&item))
            .collect::<PyResult<Vec<_>>>()
            .map(serde_json::Value::Array);
    }
    if let Ok(sequence) = value.downcast::<PyTuple>() {
        return sequence
            .iter()
            .map(|item| py_to_json(&item))
            .collect::<PyResult<Vec<_>>>()
            .map(serde_json::Value::Array);
    }
    if let Ok(dict) = value.downcast::<PyDict>() {
        let mut object = serde_json::Map::new();
        for (key, value) in dict.iter() {
            object.insert(key.extract::<String>()?, py_to_json(&value)?);
        }
        return Ok(serde_json::Value::Object(object));
    }
    Err(PyTypeError::new_err(format!(
        "cannot convert {} to JSON",
        value.get_type().name()?.to_string()
    )))
}

fn json_to_py(py: Python<'_>, value: &serde_json::Value) -> PyResult<Py<PyAny>> {
    match value {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(value) => Ok(PyBool::new(py, *value).into_any().unbind()),
        serde_json::Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                Ok(PyInt::new(py, value).into_any().unbind())
            } else if let Some(value) = value.as_u64() {
                Ok(PyInt::new(py, value).into_any().unbind())
            } else if let Some(value) = value.as_f64() {
                Ok(PyFloat::new(py, value).into_any().unbind())
            } else {
                Err(PyValueError::new_err("JSON number cannot be represented in Python"))
            }
        }
        serde_json::Value::String(value) => Ok(PyString::new(py, value).into_any().unbind()),
        serde_json::Value::Array(values) => {
            let list = PyList::empty(py);
            for value in values {
                list.append(json_to_py(py, value)?)?;
            }
            Ok(list.into_any().unbind())
        }
        serde_json::Value::Object(values) => {
            let dict = PyDict::new(py);
            for (key, value) in values {
                dict.set_item(key, json_to_py(py, value)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

fn to_json_value(value: impl serde::Serialize) -> PyResult<serde_json::Value> {
    serde_json::to_value(value).map_err(value_error)
}

fn value_error(error: impl ToString) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn runtime_error(error: impl ToString) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

enum PyFailure {
    Value(String),
    Runtime(String),
}

fn apply_failure(error: ApplyError) -> PyFailure {
    match error {
        ApplyError::Policy(error) => PyFailure::Value(error.to_string()),
        ApplyError::Conflict(conflict) => PyFailure::Value(format!("policy conflict: {conflict:?}")),
        ApplyError::InvalidResponse {
            message,
            suggestion,
        } => PyFailure::Runtime(format!("{message}\nSuggestion: {suggestion}")),
        other => PyFailure::Runtime(other.to_string()),
    }
}

fn failure_to_pyerr(error: PyFailure) -> PyErr {
    match error {
        PyFailure::Value(error) => PyValueError::new_err(error),
        PyFailure::Runtime(error) => PyRuntimeError::new_err(error),
    }
}

#[cfg(all(test, feature = "python"))]
mod tests {
    use super::*;

    fn email_policy_type() -> PolicyType {
        PolicyType {
            name: "EmailPolicy".to_string(),
            fields: vec![
                Field::Bool {
                    name: "unread".to_string(),
                    default: Some(true),
                    on_conflict: OnConflict::Default,
                },
                Field::StringEnum {
                    name: "priority".to_string(),
                    values: vec!["low".to_string(), "high".to_string()],
                    default: None,
                    on_conflict: OnConflict::LargestValue,
                },
                Field::StringArray {
                    name: "labels".to_string(),
                },
            ],
        }
    }

    fn policy_value(policy_type: &PolicyType, prompt: &str, action: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "policy_type": policy_type_to_value(policy_type),
            "prompt": prompt,
            "action": action,
        })
    }

    #[test]
    fn field_dict_round_trips() {
        let cases = vec![
            (
                Field::Bool {
                    name: "unread".to_string(),
                    default: Some(true),
                    on_conflict: OnConflict::Default,
                },
                serde_json::json!({
                    "type": "bool",
                    "name": "unread",
                    "default": true,
                    "on_conflict": "default",
                }),
            ),
            (
                Field::Number {
                    name: "score".to_string(),
                    default: Some(crate::t64(1.25)),
                    on_conflict: OnConflict::LargestValue,
                },
                serde_json::json!({
                    "type": "number",
                    "name": "score",
                    "default": 1.25,
                    "on_conflict": "largest",
                }),
            ),
            (
                Field::String {
                    name: "template".to_string(),
                    default: None,
                    on_conflict: OnConflict::Agreement,
                },
                serde_json::json!({
                    "type": "string",
                    "name": "template",
                    "default": null,
                    "on_conflict": "agreement",
                }),
            ),
            (
                Field::StringEnum {
                    name: "priority".to_string(),
                    values: vec!["low".to_string(), "high".to_string()],
                    default: Some("low".to_string()),
                    on_conflict: OnConflict::LargestValue,
                },
                serde_json::json!({
                    "type": "enum",
                    "name": "priority",
                    "values": ["low", "high"],
                    "default": "low",
                    "on_conflict": "largest",
                }),
            ),
            (
                Field::StringArray {
                    name: "labels".to_string(),
                },
                serde_json::json!({
                    "type": "string_array",
                    "name": "labels",
                }),
            ),
        ];

        for (field, value) in cases {
            assert_eq!(value, field_to_value(&field));
            assert_eq!(field, field_from_value(&value).unwrap());
        }
    }

    #[test]
    fn direct_python_json_conversion_round_trips_json_values() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let value = serde_json::json!({
                "none": null,
                "bool": true,
                "int": 42,
                "float": 1.25,
                "string": "policy",
                "array": [false, "label"],
                "object": {
                    "priority": "high",
                },
            });
            let object = json_to_py(py, &value).unwrap();

            assert_eq!(value, py_to_json(object.bind(py)).unwrap());
        });
    }

    #[test]
    fn policy_type_parse_and_default_value() {
        let policy_type = PyPolicyType::parse(
            r#"type EmailPolicy {
                unread: bool = true,
                priority: ["low", "high"] @ highest wins,
                labels: [string]
            }"#,
        )
        .unwrap();

        assert_eq!(email_policy_type(), policy_type.inner);
        assert_eq!(serde_json::json!({"unread": true}), policy_type.inner.default_value());
    }

    #[test]
    fn policy_rejects_missing_action() {
        let value = serde_json::json!({
            "policy_type": policy_type_to_value(&email_policy_type()),
            "prompt": "If urgent, set priority high.",
        });

        let error = policy_from_value(&value).unwrap_err();
        assert!(error.to_string().contains("policy dict must include 'action'"));
    }

    #[test]
    fn manager_rejects_mismatched_policy_type() {
        let email = email_policy_type();
        let task = PolicyType {
            name: "TaskPolicy".to_string(),
            fields: vec![Field::Bool {
                name: "done".to_string(),
                default: Some(false),
                on_conflict: OnConflict::Default,
            }],
        };
        let mut manager = Manager::default();
        manager
            .try_add(policy_from_value(&policy_value(
                &email,
                "If urgent, set priority high.",
                serde_json::json!({"priority": "high"}),
            ))
            .unwrap())
            .unwrap();
        let error = manager
            .try_add(
                policy_from_value(&policy_value(
                    &task,
                    "If complete, mark done.",
                    serde_json::json!({"done": true}),
                ))
                .unwrap(),
            )
            .unwrap_err();

        assert!(error.to_string().contains("Policy type mismatch"));
    }

    #[test]
    fn parse_on_conflict_aliases() {
        assert_eq!(OnConflict::Default, parse_on_conflict("default").unwrap());
        assert_eq!(OnConflict::Agreement, parse_on_conflict("agreement").unwrap());
        assert_eq!(OnConflict::Agreement, parse_on_conflict("must-agree").unwrap());
        assert_eq!(OnConflict::LargestValue, parse_on_conflict("largest").unwrap());
        assert_eq!(OnConflict::LargestValue, parse_on_conflict("sticky").unwrap());
        assert_eq!(OnConflict::LargestValue, parse_on_conflict("highest wins").unwrap());
        assert!(parse_on_conflict("nonsense").is_err());
    }

    #[test]
    fn inference_config_aliases() {
        assert_eq!(
            InferenceConfig::OutputFormatJsonSchema,
            parse_inference_config("json_schema").unwrap()
        );
        assert_eq!(
            InferenceConfig::OutputFormatJsonSchema,
            parse_inference_config("output-format").unwrap()
        );
        assert_eq!(InferenceConfig::ToolUse, parse_inference_config("tool_use").unwrap());
        assert_eq!(
            InferenceConfig::StrictToolUse,
            parse_inference_config("strict-tool-use").unwrap()
        );
        assert!(parse_inference_config("nonsense").is_err());
    }

    #[test]
    fn report_to_dict_value_is_complete() {
        let mut report = Report::new(vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
        report.report_bool_default("active", true);
        report.report_bool(1, "active", false, OnConflict::Default);

        assert_eq!(
            serde_json::json!({
                "value": {"active": false},
                "ir": null,
                "default": {"active": true},
                "rules_matched": [1],
                "errors": [],
                "conflicts": [],
                "field_stats": {
                    "active": {
                        "count": 1,
                        "distribution": {
                            "false": 1,
                        },
                    },
                },
                "usage": null,
            }),
            report_to_value(&report, None).unwrap()
        );
    }
}
