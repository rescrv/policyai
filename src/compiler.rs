//! Compile line-oriented policy declarations into PolicyAI policies.
//!
//! The compiler accepts the compact policy declaration format used by tools that
//! emit one semantic injection and one JSON action per line:
//!
//! ```text
//! If the message is urgent {"priority": "high"}
//! If the message is spam {"archive": true}
//! ```
//!
//! The caller supplies the [`PolicyType`](crate::PolicyType).  The compiler
//! parses each line, validates the JSON action against that type, and can then
//! produce [`Policy`](crate::Policy) values or a [`Manager`](crate::Manager).

use crate::{Field, Manager, Policy, PolicyError, PolicyType};

/// A parsed semantic injection paired with the JSON action it should trigger.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct InjectionAction {
    /// The natural-language condition or instruction for this policy rule.
    pub injection: String,
    /// The structured JSON action to apply when the injection matches.
    pub action: serde_json::Value,
}

impl InjectionAction {
    /// Create a new injection/action pair.
    pub fn new(injection: impl Into<String>, action: serde_json::Value) -> Self {
        Self {
            injection: injection.into(),
            action,
        }
    }

    /// Convert this pair into a [`Policy`] for the supplied policy type.
    pub fn into_policy(self, r#type: PolicyType) -> Policy {
        Policy {
            r#type,
            prompt: self.injection,
            action: self.action,
        }
    }

    /// Clone this pair into a [`Policy`] for the supplied policy type.
    pub fn to_policy(&self, r#type: PolicyType) -> Policy {
        Policy {
            r#type,
            prompt: self.injection.clone(),
            action: self.action.clone(),
        }
    }
}

impl From<crate::data::InjectableAction> for InjectionAction {
    fn from(value: crate::data::InjectableAction) -> Self {
        Self {
            injection: value.inject,
            action: value.action,
        }
    }
}

impl From<InjectionAction> for crate::data::InjectableAction {
    fn from(value: InjectionAction) -> Self {
        Self {
            inject: value.injection,
            action: value.action,
        }
    }
}

impl std::str::FromStr for InjectionAction {
    type Err = CompileError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parse_injection_action(s)
    }
}

/// A validated set of policy declarations bound to a single policy type.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct CompiledPolicySet {
    /// The policy type all compiled policies use.
    pub r#type: PolicyType,
    /// The parsed and validated injection/action pairs.
    pub injections: Vec<InjectionAction>,
}

impl CompiledPolicySet {
    /// Create a compiled policy set from pre-parsed injection/action pairs.
    ///
    /// This validates every action against `type_`.
    ///
    /// # Errors
    ///
    /// Returns [`CompileError`] if an action is not a JSON object, references an
    /// unknown field, or provides a value incompatible with the field type.
    pub fn new(type_: PolicyType, injections: Vec<InjectionAction>) -> Result<Self, CompileError> {
        for injection in &injections {
            validate_action(&type_, &injection.action, None)?;
        }
        Ok(Self {
            r#type: type_,
            injections,
        })
    }

    /// Create a compiled policy set and validate actions as a Rust type.
    ///
    /// This validates every action against `type_` and also verifies that each
    /// action deserializes as `T`.
    ///
    /// # Errors
    ///
    /// Returns [`CompileError`] if an action is invalid for the policy type or
    /// cannot be deserialized as `T`.
    pub fn new_with_action_type<T>(
        type_: PolicyType,
        injections: Vec<InjectionAction>,
    ) -> Result<Self, CompileError>
    where
        T: serde::de::DeserializeOwned,
    {
        for injection in &injections {
            validate_action(&type_, &injection.action, None)?;
            validate_action_type::<T>(&injection.action, None)?;
        }
        Ok(Self {
            r#type: type_,
            injections,
        })
    }

    /// Parse and validate compiler output for a caller-provided policy type.
    ///
    /// Empty lines are ignored.
    ///
    /// # Errors
    ///
    /// Returns [`CompileError`] if any non-empty line cannot be parsed or any
    /// action is invalid for `type_`.
    pub fn parse(type_: PolicyType, input: &str) -> Result<Self, CompileError> {
        compile(type_, input)
    }

    /// Parse compiler output and validate actions as a Rust type.
    ///
    /// Empty lines are ignored.
    ///
    /// # Errors
    ///
    /// Returns [`CompileError`] if any line cannot be parsed, any action is
    /// invalid for `type_`, or any action cannot be deserialized as `T`.
    pub fn parse_with_action_type<T>(type_: PolicyType, input: &str) -> Result<Self, CompileError>
    where
        T: serde::de::DeserializeOwned,
    {
        compile_with_action_type::<T>(type_, input)
    }

    /// Return the number of compiled policy declarations.
    pub fn len(&self) -> usize {
        self.injections.len()
    }

    /// Return `true` when there are no compiled policy declarations.
    pub fn is_empty(&self) -> bool {
        self.injections.is_empty()
    }

    /// Iterate over the parsed injection/action pairs.
    pub fn iter(&self) -> impl Iterator<Item = &InjectionAction> {
        self.injections.iter()
    }

    /// Clone the compiled declarations into PolicyAI [`Policy`] values.
    pub fn policies(&self) -> Vec<Policy> {
        self.injections
            .iter()
            .map(|injection| injection.to_policy(self.r#type.clone()))
            .collect()
    }

    /// Consume the compiled declarations into PolicyAI [`Policy`] values.
    pub fn into_policies(self) -> Vec<Policy> {
        let type_ = self.r#type;
        self.injections
            .into_iter()
            .map(|injection| injection.into_policy(type_.clone()))
            .collect()
    }

    /// Add the compiled policies to an existing [`Manager`].
    ///
    /// # Errors
    ///
    /// Returns [`PolicyError::PolicyTypeMismatch`] if `manager` already contains
    /// policies for a different policy type.
    #[allow(clippy::result_large_err)]
    pub fn extend_manager(&self, manager: &mut Manager) -> Result<(), PolicyError> {
        for policy in self.policies() {
            manager.add(policy)?;
        }
        Ok(())
    }

    /// Consume this set and construct a [`Manager`] containing every policy.
    ///
    /// This cannot fail because every compiled policy uses the same policy type.
    pub fn into_manager(self) -> Manager {
        let mut manager = Manager::default();
        for policy in self.into_policies() {
            manager
                .add(policy)
                .expect("compiled policies always share one policy type");
        }
        manager
    }
}

impl From<CompiledPolicySet> for Manager {
    fn from(value: CompiledPolicySet) -> Self {
        value.into_manager()
    }
}

/// A compiler error with optional source line context.
#[derive(Clone, Debug, PartialEq)]
pub struct CompileError {
    line: Option<usize>,
    kind: CompileErrorKind,
}

impl CompileError {
    /// Create an error without line context.
    pub fn new(kind: CompileErrorKind) -> Self {
        Self { line: None, kind }
    }

    /// Create an error with one-based line context.
    pub fn with_line(line: usize, kind: CompileErrorKind) -> Self {
        Self {
            line: Some(line),
            kind,
        }
    }

    /// Return the one-based line number, if this error came from multi-line input.
    pub fn line(&self) -> Option<usize> {
        self.line
    }

    /// Return the underlying error kind.
    pub fn kind(&self) -> &CompileErrorKind {
        &self.kind
    }
}

/// The specific reason compilation failed.
#[derive(Clone, Debug, PartialEq)]
pub enum CompileErrorKind {
    /// The requested line is empty or contains only whitespace.
    EmptyLine,
    /// No valid JSON value could be found at the end of the line.
    NoValidJson,
    /// The action JSON must be an object keyed by policy field name.
    ActionNotObject {
        /// The JSON type that was provided.
        actual: String,
    },
    /// The action object contains a field not declared by the policy type.
    UnknownField {
        /// The undeclared field name.
        field: String,
    },
    /// A field value has the wrong JSON type.
    InvalidFieldValue {
        /// The policy field name.
        field: String,
        /// The expected value description.
        expected: String,
        /// The actual JSON type that was provided.
        actual: String,
    },
    /// A string enum field was assigned a string outside the declared values.
    InvalidEnumValue {
        /// The policy field name.
        field: String,
        /// The value that was provided.
        value: String,
        /// The declared enum values.
        allowed: Vec<String>,
    },
    /// The action could not be deserialized as the caller-provided Rust type.
    InvalidActionType {
        /// The Rust type name supplied by the caller.
        type_name: String,
        /// The deserialization error.
        error: String,
    },
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(line) = self.line {
            write!(f, "line {line}: ")?;
        }
        match &self.kind {
            CompileErrorKind::EmptyLine => write!(f, "empty line cannot be parsed"),
            CompileErrorKind::NoValidJson => write!(f, "no valid JSON found at end of line"),
            CompileErrorKind::ActionNotObject { actual } => {
                write!(f, "policy action must be a JSON object, got {actual}")
            }
            CompileErrorKind::UnknownField { field } => {
                write!(f, "unknown policy field {field:?}")
            }
            CompileErrorKind::InvalidFieldValue {
                field,
                expected,
                actual,
            } => write!(
                f,
                "invalid value for field {field:?}: expected {expected}, got {actual}"
            ),
            CompileErrorKind::InvalidEnumValue {
                field,
                value,
                allowed,
            } => write!(
                f,
                "invalid enum value for field {field:?}: got {value:?}, expected one of {}",
                allowed
                    .iter()
                    .map(|value| format!("{value:?}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            CompileErrorKind::InvalidActionType { type_name, error } => {
                write!(f, "action does not deserialize as {type_name}: {error}")
            }
        }
    }
}

impl std::error::Error for CompileError {}

/// Parse one declaration of the form `injection JSON`.
///
/// The parser finds the longest valid JSON suffix and treats everything before
/// it as the semantic injection text.
///
/// # Errors
///
/// Returns [`CompileErrorKind::EmptyLine`] for blank input and
/// [`CompileErrorKind::NoValidJson`] when no JSON suffix can be parsed.
pub fn parse_injection_action(line: &str) -> Result<InjectionAction, CompileError> {
    parse_injection_action_inner(line, None)
}

/// Parse every non-empty line of compiler output into injection/action pairs.
///
/// Empty lines are ignored.  Errors include one-based line numbers.
///
/// # Errors
///
/// Returns [`CompileError`] when any non-empty line lacks a valid JSON suffix.
pub fn parse_injection_actions(input: &str) -> Result<Vec<InjectionAction>, CompileError> {
    let mut injections = Vec::new();
    for (index, line) in input.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        injections.push(parse_injection_action_inner(line, Some(index + 1))?);
    }
    Ok(injections)
}

/// Parse and validate compiler output for a caller-provided policy type.
///
/// This is the main entry point when loading declarations such as
/// `injection {"action": "to_take"}` from a file or model output.
///
/// # Errors
///
/// Returns [`CompileError`] when parsing fails or an action is invalid for
/// `type_`.
pub fn compile(type_: PolicyType, input: &str) -> Result<CompiledPolicySet, CompileError> {
    compile_with_validator(type_, input, |_, _| Ok(()))
}

/// Parse compiler output and validate each action as a caller-provided Rust type.
///
/// This mirrors applications that want to validate actions with their own
/// `serde::Deserialize` type, while still producing ordinary PolicyAI policies.
///
/// # Errors
///
/// Returns [`CompileError`] when parsing fails, an action is invalid for
/// `type_`, or an action cannot be deserialized as `T`.
pub fn compile_with_action_type<T>(
    type_: PolicyType,
    input: &str,
) -> Result<CompiledPolicySet, CompileError>
where
    T: serde::de::DeserializeOwned,
{
    compile_with_validator(type_, input, validate_action_type::<T>)
}

fn compile_with_validator(
    type_: PolicyType,
    input: &str,
    validate_extra: impl Fn(&serde_json::Value, Option<usize>) -> Result<(), CompileError>,
) -> Result<CompiledPolicySet, CompileError> {
    let mut injections = Vec::new();
    for (index, line) in input.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let line_number = index + 1;
        let injection = parse_injection_action_inner(line, Some(line_number))?;
        validate_action(&type_, &injection.action, Some(line_number))?;
        validate_extra(&injection.action, Some(line_number))?;
        injections.push(injection);
    }
    Ok(CompiledPolicySet {
        r#type: type_,
        injections,
    })
}

/// Parse compiler output directly into PolicyAI [`Policy`] values.
///
/// # Errors
///
/// Returns [`CompileError`] when parsing or validation fails.
pub fn compile_policies(type_: PolicyType, input: &str) -> Result<Vec<Policy>, CompileError> {
    Ok(compile(type_, input)?.into_policies())
}

/// Parse compiler output directly into [`Policy`] values with Rust type validation.
///
/// # Errors
///
/// Returns [`CompileError`] when parsing or validation fails.
pub fn compile_policies_with_action_type<T>(
    type_: PolicyType,
    input: &str,
) -> Result<Vec<Policy>, CompileError>
where
    T: serde::de::DeserializeOwned,
{
    Ok(compile_with_action_type::<T>(type_, input)?.into_policies())
}

/// Parse compiler output directly into a PolicyAI [`Manager`].
///
/// # Errors
///
/// Returns [`CompileError`] when parsing or validation fails.
pub fn compile_manager(type_: PolicyType, input: &str) -> Result<Manager, CompileError> {
    Ok(compile(type_, input)?.into_manager())
}

/// Parse compiler output directly into a [`Manager`] with Rust type validation.
///
/// # Errors
///
/// Returns [`CompileError`] when parsing or validation fails.
pub fn compile_manager_with_action_type<T>(
    type_: PolicyType,
    input: &str,
) -> Result<Manager, CompileError>
where
    T: serde::de::DeserializeOwned,
{
    Ok(compile_with_action_type::<T>(type_, input)?.into_manager())
}

fn parse_injection_action_inner(
    line: &str,
    line_number: Option<usize>,
) -> Result<InjectionAction, CompileError> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Err(error(line_number, CompileErrorKind::EmptyLine));
    }

    let mut longest_match = None;
    for (start, _) in trimmed.char_indices().rev() {
        let candidate = &trimmed[start..];
        if let Ok(action) = serde_json::from_str::<serde_json::Value>(candidate) {
            longest_match = Some((start, action));
        }
    }

    if let Some((start, action)) = longest_match {
        let injection = trimmed[..start].trim_end().to_string();
        return Ok(InjectionAction { injection, action });
    }

    Err(error(line_number, CompileErrorKind::NoValidJson))
}

fn validate_action(
    type_: &PolicyType,
    action: &serde_json::Value,
    line_number: Option<usize>,
) -> Result<(), CompileError> {
    let Some(object) = action.as_object() else {
        return Err(error(
            line_number,
            CompileErrorKind::ActionNotObject {
                actual: json_type(action).to_string(),
            },
        ));
    };

    for (name, value) in object {
        let Some(field) = type_.fields.iter().find(|field| field.name() == name) else {
            return Err(error(
                line_number,
                CompileErrorKind::UnknownField {
                    field: name.clone(),
                },
            ));
        };
        validate_field_value(field, value, line_number)?;
    }

    Ok(())
}

fn validate_action_type<T>(
    action: &serde_json::Value,
    line_number: Option<usize>,
) -> Result<(), CompileError>
where
    T: serde::de::DeserializeOwned,
{
    serde_json::from_value::<T>(action.clone())
        .map(|_| ())
        .map_err(|err| {
            error(
                line_number,
                CompileErrorKind::InvalidActionType {
                    type_name: std::any::type_name::<T>().to_string(),
                    error: err.to_string(),
                },
            )
        })
}

fn validate_field_value(
    field: &Field,
    value: &serde_json::Value,
    line_number: Option<usize>,
) -> Result<(), CompileError> {
    match field {
        Field::Bool { name, .. } if !value.is_boolean() => Err(invalid_field_value(
            line_number,
            name,
            "bool",
            json_type(value),
        )),
        Field::Number { name, .. } if !value.is_number() && !value.is_null() => Err(
            invalid_field_value(line_number, name, "number or null", json_type(value)),
        ),
        Field::String { name, .. } if !value.is_string() && !value.is_null() => Err(
            invalid_field_value(line_number, name, "string or null", json_type(value)),
        ),
        Field::StringEnum { name, values, .. } => {
            if value.is_null() {
                Ok(())
            } else if let Some(value) = value.as_str() {
                if values.iter().any(|allowed| allowed == value) {
                    Ok(())
                } else {
                    Err(error(
                        line_number,
                        CompileErrorKind::InvalidEnumValue {
                            field: name.clone(),
                            value: value.to_string(),
                            allowed: values.clone(),
                        },
                    ))
                }
            } else {
                Err(invalid_field_value(
                    line_number,
                    name,
                    "one of the declared enum strings or null",
                    json_type(value),
                ))
            }
        }
        Field::StringArray { name } => {
            if value
                .as_array()
                .is_some_and(|values| values.iter().all(serde_json::Value::is_string))
            {
                Ok(())
            } else {
                Err(invalid_field_value(
                    line_number,
                    name,
                    "array of strings",
                    json_type(value),
                ))
            }
        }
        _ => Ok(()),
    }
}

fn invalid_field_value(
    line_number: Option<usize>,
    field: &str,
    expected: &str,
    actual: &str,
) -> CompileError {
    error(
        line_number,
        CompileErrorKind::InvalidFieldValue {
            field: field.to_string(),
            expected: expected.to_string(),
            actual: actual.to_string(),
        },
    )
}

fn error(line_number: Option<usize>, kind: CompileErrorKind) -> CompileError {
    match line_number {
        Some(line) => CompileError::with_line(line, kind),
        None => CompileError::new(kind),
    }
}

fn json_type(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OnConflict;

    fn policy_type() -> PolicyType {
        PolicyType {
            name: "TestPolicy".to_string(),
            fields: vec![
                Field::Bool {
                    name: "enabled".to_string(),
                    default: Some(false),
                    on_conflict: OnConflict::Default,
                },
                Field::Number {
                    name: "score".to_string(),
                    default: None,
                    on_conflict: OnConflict::LargestValue,
                },
                Field::String {
                    name: "note".to_string(),
                    default: None,
                    on_conflict: OnConflict::Agreement,
                },
                Field::StringEnum {
                    name: "action".to_string(),
                    values: vec!["skip".to_string(), "to_take".to_string()],
                    default: None,
                    on_conflict: OnConflict::LargestValue,
                },
                Field::StringArray {
                    name: "labels".to_string(),
                },
            ],
        }
    }

    #[derive(Debug, serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Decision {
        action: String,
        #[serde(default)]
        labels: Vec<String>,
    }

    #[test]
    fn parse_one_injection_action() {
        let parsed = parse_injection_action(r#"injection {"action": "to_take"}"#).unwrap();

        assert_eq!("injection", parsed.injection);
        assert_eq!(serde_json::json!({"action": "to_take"}), parsed.action);
    }

    #[test]
    fn parse_uses_longest_valid_json_suffix() {
        let parsed =
            parse_injection_action(r#"Found {old: value} in text {"note": "new"}"#).unwrap();

        assert_eq!("Found {old: value} in text", parsed.injection);
        assert_eq!(serde_json::json!({"note": "new"}), parsed.action);
    }

    #[test]
    fn parse_multiple_lines_ignores_empty_lines() {
        let parsed = parse_injection_actions(
            r#"
enabled rule {"enabled": true}

score rule {"score": 1.5}
"#,
        )
        .unwrap();

        assert_eq!(2, parsed.len());
        assert_eq!("enabled rule", parsed[0].injection);
        assert_eq!(serde_json::json!({"score": 1.5}), parsed[1].action);
    }

    #[test]
    fn parse_error_has_line_number() {
        let err = parse_injection_actions("valid {\"enabled\": true}\ninvalid").unwrap_err();

        assert_eq!(Some(2), err.line());
        assert_eq!(&CompileErrorKind::NoValidJson, err.kind());
        assert_eq!(
            "line 2: no valid JSON found at end of line",
            err.to_string()
        );
    }

    #[test]
    fn compile_binds_to_policy_type() {
        let compiled = compile(
            policy_type(),
            r#"
injection {"action": "to_take"}
other {"enabled": true, "score": 2, "note": null, "labels": ["a", "b"]}
"#,
        )
        .unwrap();

        assert_eq!(2, compiled.len());
        let policies = compiled.policies();
        assert_eq!("TestPolicy", policies[0].r#type.name);
        assert_eq!("injection", policies[0].prompt);
        assert_eq!(serde_json::json!({"action": "to_take"}), policies[0].action);
        assert_eq!(2, policies.len());
    }

    #[test]
    fn compile_with_action_type_accepts_user_type() {
        let compiled = compile_with_action_type::<Decision>(
            policy_type(),
            r#"injection {"action": "to_take", "labels": ["x"]}"#,
        )
        .unwrap();

        let decision: Decision =
            serde_json::from_value(compiled.injections[0].action.clone()).unwrap();
        assert_eq!("to_take", decision.action);
        assert_eq!(vec!["x".to_string()], decision.labels);
    }

    #[test]
    fn compile_with_action_type_rejects_user_type_mismatch() {
        let err = compile_with_action_type::<Decision>(policy_type(), r#"bad {"enabled": true}"#)
            .unwrap_err();

        assert_eq!(Some(1), err.line());
        let CompileErrorKind::InvalidActionType { type_name, error } = err.kind() else {
            panic!("expected InvalidActionType");
        };
        assert!(type_name.contains("Decision"));
        assert!(error.contains("unknown field"));
    }

    #[test]
    fn compile_rejects_non_object_actions() {
        let err = compile(policy_type(), "bad true").unwrap_err();

        assert_eq!(Some(1), err.line());
        assert_eq!(
            &CompileErrorKind::ActionNotObject {
                actual: "bool".to_string()
            },
            err.kind()
        );
    }

    #[test]
    fn compile_rejects_unknown_fields() {
        let err = compile(policy_type(), r#"bad {"unknown": true}"#).unwrap_err();

        assert_eq!(
            &CompileErrorKind::UnknownField {
                field: "unknown".to_string()
            },
            err.kind()
        );
    }

    #[test]
    fn compile_rejects_wrong_field_types() {
        let err = compile(policy_type(), r#"bad {"enabled": "true"}"#).unwrap_err();

        assert_eq!(
            &CompileErrorKind::InvalidFieldValue {
                field: "enabled".to_string(),
                expected: "bool".to_string(),
                actual: "string".to_string()
            },
            err.kind()
        );
    }

    #[test]
    fn compile_rejects_invalid_enum_values() {
        let err = compile(policy_type(), r#"bad {"action": "archive"}"#).unwrap_err();

        assert_eq!(
            &CompileErrorKind::InvalidEnumValue {
                field: "action".to_string(),
                value: "archive".to_string(),
                allowed: vec!["skip".to_string(), "to_take".to_string()]
            },
            err.kind()
        );
    }

    #[test]
    fn compiled_policy_set_constructs_manager() {
        let manager = compile_manager(
            policy_type(),
            r#"
enabled {"enabled": true}
action {"action": "to_take"}
"#,
        )
        .unwrap();

        assert_eq!(2, manager.len());
    }

    #[test]
    fn pre_parsed_injections_can_be_validated() {
        let injections = vec![InjectionAction::new(
            "action",
            serde_json::json!({"action": "to_take"}),
        )];
        let compiled = CompiledPolicySet::new(policy_type(), injections).unwrap();

        assert_eq!(1, compiled.len());
    }
}
