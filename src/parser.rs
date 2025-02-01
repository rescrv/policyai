use std::fmt::{Formatter, Write};

use nom::{
    branch::alt,
    bytes::complete::{escaped, tag},
    character::complete::{alpha1, alphanumeric1, digit1, multispace0, none_of, one_of},
    combinator::{all_consuming, cut, map, map_res, opt, recognize},
    error::{context, VerboseError, VerboseErrorKind},
    multi::{many0_count, separated_list1},
    sequence::{delimited, pair, terminated, tuple},
    IResult, Offset,
};

use crate::{t64, Field, OnConflict, PolicyType};

////////////////////////////////////////// error handling //////////////////////////////////////////

type ParseResult<'a, T> = IResult<&'a str, T, VerboseError<&'a str>>;

#[derive(Clone, Eq, PartialEq)]
pub struct ParseError {
    string: String,
}

impl std::fmt::Debug for ParseError {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        writeln!(fmt, "{}", self.string)
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        writeln!(fmt, "{}", self.string)
    }
}

impl From<String> for ParseError {
    fn from(string: String) -> Self {
        Self { string }
    }
}

pub fn interpret_verbose_error(input: &'_ str, err: VerboseError<&'_ str>) -> ParseError {
    let mut result = String::new();
    let mut index = 0;
    for (substring, kind) in err.errors.iter() {
        let offset = input.offset(substring);
        let prefix = &input.as_bytes()[..offset];
        // Count the number of newlines in the first `offset` bytes of input
        let line_number = prefix.iter().filter(|&&b| b == b'\n').count() + 1;
        // Find the line that includes the subslice:
        // Find the *last* newline before the substring starts
        let line_begin = prefix
            .iter()
            .rev()
            .position(|&b| b == b'\n')
            .map(|pos| offset - pos)
            .unwrap_or(0);
        // Find the full line after that newline
        let line = input[line_begin..]
            .lines()
            .next()
            .unwrap_or(&input[line_begin..])
            .trim_end();
        // The (1-indexed) column number is the offset of our substring into that line
        let column_number = line.offset(substring) + 1;
        match kind {
            VerboseErrorKind::Char(c) => {
                if let Some(actual) = substring.chars().next() {
                    write!(
                        &mut result,
                        "{index}: at line {line_number}:\n\
                 {line}\n\
                 {caret:>column$}\n\
                 expected '{expected}', found {actual}\n\n",
                        index = index,
                        line_number = line_number,
                        line = line,
                        caret = '^',
                        column = column_number,
                        expected = c,
                        actual = actual,
                    )
                    .unwrap();
                } else {
                    write!(
                        &mut result,
                        "{index}: at line {line_number}:\n\
                 {line}\n\
                 {caret:>column$}\n\
                 expected '{expected}', got end of input\n\n",
                        index = index,
                        line_number = line_number,
                        line = line,
                        caret = '^',
                        column = column_number,
                        expected = c,
                    )
                    .unwrap();
                }
                index += 1;
            }
            VerboseErrorKind::Context(s) => {
                write!(
                    &mut result,
                    "{index}: at line {line_number}, in {context}:\n\
               {line}\n\
               {caret:>column$}\n\n",
                    index = index,
                    line_number = line_number,
                    context = s,
                    line = line,
                    caret = '^',
                    column = column_number,
                )
                .unwrap();
                index += 1;
            }
            // Swallow these.   They are ugly.
            VerboseErrorKind::Nom(_) => {}
        };
    }
    ParseError {
        string: result.trim().to_string(),
    }
}

///////////////////////////////////////////// TableSet /////////////////////////////////////////////

pub fn identifier(input: &str) -> ParseResult<String> {
    context(
        "identifier",
        map(
            recognize(pair(
                alt((alpha1, tag("_"))),
                many0_count(alt((alphanumeric1, tag("_")))),
            )),
            |ident: &str| ident.to_string(),
        ),
    )(input)
}

pub fn unescape(input: &str) -> String {
    let mut out: Vec<char> = Vec::new();
    let mut prev_was_escape = false;
    for c in input.chars() {
        if prev_was_escape && (c == '\"' || c == '\\') {
            out.push(c);
            prev_was_escape = false;
        } else if c == '\\' {
            prev_was_escape = true;
        } else {
            out.push(c);
        }
    }
    out.into_iter().collect()
}

pub fn string_literal(input: &str) -> ParseResult<String> {
    context(
        "string literal",
        map(
            delimited(
                tag("\""),
                cut(alt((
                    escaped(none_of(r#"\""#), '\\', one_of(r#"\""#)),
                    tag(""),
                ))),
                tag("\""),
            ),
            |x: &str| unescape(x),
        ),
    )(input)
}

pub fn number_literal(input: &str) -> ParseResult<f64> {
    // TODO(rescrv):  Make this support float.
    context(
        "number literal",
        map_res(recognize(tuple((opt(tag("-")), digit1))), str::parse::<f64>),
    )(input)
}

pub fn bool_conflicts(input: &str) -> ParseResult<OnConflict> {
    context(
        "bool conflicts",
        map(
            opt(map(tuple((ws0, tag("@"), ws0, tag("agreement"))), |_| {
                OnConflict::Agreement
            })),
            |x| x.unwrap_or_default(),
        ),
    )(input)
}

pub fn optional_default_bool(input: &str) -> ParseResult<bool> {
    context(
        "bool default",
        map(
            opt(alt((
                map(tuple((ws0, tag("="), ws0, tag("true"))), |_| true),
                map(tuple((ws0, tag("="), ws0, tag("false"))), |_| false),
            ))),
            |b| b.unwrap_or(false),
        ),
    )(input)
}

pub fn string_conflicts(input: &str) -> ParseResult<OnConflict> {
    context(
        "string conflicts",
        map(
            opt(map(tuple((ws0, tag("@"), ws0, tag("agreement"))), |_| {
                OnConflict::Agreement
            })),
            |x| x.unwrap_or_default(),
        ),
    )(input)
}

pub fn optional_default_string(input: &str) -> ParseResult<Option<String>> {
    context(
        "string default",
        opt(map(
            tuple((ws0, tag("="), ws0, string_literal)),
            |(_, _, _, x)| x,
        )),
    )(input)
}

pub fn string_enum_conflicts(input: &str) -> ParseResult<OnConflict> {
    context(
        "string enum conflicts",
        map(
            opt(alt((
                map(
                    tuple((ws0, tag("@"), ws0, tag("highest"), ws0, tag("wins"))),
                    |_| OnConflict::LargestValue,
                ),
                map(tuple((ws0, tag("@"), ws0, tag("agreement"))), |_| {
                    OnConflict::Agreement
                }),
            ))),
            |x| x.unwrap_or_default(),
        ),
    )(input)
}

pub fn optional_default_string_enum(input: &str) -> ParseResult<Option<String>> {
    context(
        "string enum default",
        opt(map(
            tuple((ws0, tag("="), ws0, string_literal)),
            |(_, _, _, x)| x,
        )),
    )(input)
}

pub fn number_conflicts(input: &str) -> ParseResult<OnConflict> {
    context(
        "number conflicts",
        map(
            opt(alt((
                map(
                    tuple((ws0, tag("@"), ws0, tag("largest"), ws0, tag("wins"))),
                    |_| OnConflict::LargestValue,
                ),
                map(tuple((ws0, tag("@"), ws0, tag("agreement"))), |_| {
                    OnConflict::Agreement
                }),
            ))),
            |x| x.unwrap_or_default(),
        ),
    )(input)
}

pub fn optional_default_number(input: &str) -> ParseResult<Option<f64>> {
    context(
        "number default",
        opt(map(
            tuple((ws0, tag("="), ws0, number_literal)),
            |(_, _, _, x)| x,
        )),
    )(input)
}

pub fn field(input: &str) -> ParseResult<Field> {
    context(
        "field",
        alt((
            map(
                tuple((
                    ws0,
                    identifier,
                    ws0,
                    tag(":"),
                    ws0,
                    tag("bool"),
                    cut(ws0),
                    bool_conflicts,
                    ws0,
                    optional_default_bool,
                )),
                |(_, name, _, _, _, _, _, on_conflict, _, default)| Field::Bool {
                    name,
                    on_conflict,
                    default,
                },
            ),
            map(
                tuple((
                    ws0,
                    identifier,
                    ws0,
                    tag(":"),
                    ws0,
                    tag("string"),
                    cut(ws0),
                    string_conflicts,
                    ws0,
                    optional_default_string,
                )),
                |(_, name, _, _, _, _, _, on_conflict, _, default)| Field::String {
                    name,
                    on_conflict,
                    default,
                },
            ),
            map(
                tuple((
                    ws0,
                    identifier,
                    ws0,
                    tag(":"),
                    ws0,
                    tag("[string]"),
                    cut(ws0),
                )),
                |(_, name, _, _, _, _, _)| Field::StringArray { name },
            ),
            map(
                tuple((
                    ws0,
                    identifier,
                    ws0,
                    tag(":"),
                    ws0,
                    tag("["),
                    ws0,
                    separated_list1(tuple((ws0, tag(","), ws0)), string_literal),
                    ws0,
                    tag("]"),
                    ws0,
                    string_enum_conflicts,
                    ws0,
                    optional_default_string_enum,
                )),
                |(_, name, _, _, _, _, _, values, _, _, _, on_conflict, _, default)| {
                    Field::StringEnum {
                        name,
                        values,
                        on_conflict,
                        default,
                    }
                },
            ),
            map(
                tuple((
                    ws0,
                    identifier,
                    ws0,
                    tag(":"),
                    ws0,
                    tag("number"),
                    ws0,
                    number_conflicts,
                    ws0,
                    optional_default_number,
                )),
                |(_, name, _, _, _, _, _, on_conflict, _, default)| Field::Number {
                    name,
                    on_conflict,
                    default: default.map(t64),
                },
            ),
        )),
    )(input)
}

pub fn policy_type(input: &str) -> ParseResult<PolicyType> {
    context(
        "policy type",
        map(
            tuple((
                ws0,
                tag("type"),
                ws0,
                separated_list1(tag("::"), identifier),
                ws0,
                tag("{"),
                ws0,
                terminated(separated_list1(tag(","), field), opt(tag(","))),
                ws0,
                tag("}"),
                ws0,
            )),
            |(_, _, _, name, _, _, _, fields, _, _, _)| PolicyType {
                name: name.join("::"),
                fields,
            },
        ),
    )(input)
}

///////////////////////////////////////////// parse_all ////////////////////////////////////////////

pub fn parse_all<T, F: Fn(&str) -> ParseResult<T> + Copy>(
    f: F,
) -> impl Fn(&str) -> Result<T, ParseError> {
    move |input| {
        let (rem, t) = match all_consuming(f)(input) {
            Ok((rem, t)) => (rem, t),
            Err(err) => match err {
                nom::Err::Incomplete(_) => {
                    panic!("all_consuming combinator should be all consuming");
                }
                nom::Err::Error(err) | nom::Err::Failure(err) => {
                    return Err(interpret_verbose_error(input, err));
                }
            },
        };
        if rem.is_empty() {
            Ok(t)
        } else {
            panic!("all_consuming combinator should be all consuming");
        }
    }
}

////////////////////////////////////////////// private /////////////////////////////////////////////

fn ws0(input: &str) -> ParseResult<()> {
    map(multispace0, |_| ())(input)
}

/////////////////////////////////////////////// tests //////////////////////////////////////////////

#[cfg(test)]
mod test {
    use nom::combinator::{complete, cut};

    use super::*;

    fn parse_error(s: &'static str) -> ParseError {
        ParseError {
            string: s.to_string(),
        }
    }

    fn interpret_error_for_test<'a, T, F: FnMut(&'a str) -> ParseResult<T>>(
        mut f: F,
    ) -> impl FnMut(&'a str) -> Result<T, ParseError> {
        move |input| match f(input) {
            Ok((_, t)) => Ok(t),
            Err(err) => match err {
                nom::Err::Error(err) | nom::Err::Failure(err) => {
                    Err(interpret_verbose_error(input, err))
                }
                nom::Err::Incomplete(_) => {
                    panic!("incomplete should never happen in tests");
                }
            },
        }
    }

    #[test]
    fn identifier9() {
        assert_eq!(
            "__identifier9",
            parse_all(identifier)("__identifier9").unwrap(),
        );
    }

    #[test]
    fn identifier_empty() {
        assert_eq!(
            parse_error(
                r#"0: at line 1, in identifier:

^"#
            ),
            interpret_error_for_test(cut(complete(all_consuming(identifier))))("").unwrap_err()
        );
    }

    #[test]
    fn identifier_dashes() {
        assert_eq!(
            parse_error(
                r#"0: at line 1, in identifier:
-not-identifier
^"#
            ),
            interpret_error_for_test(cut(complete(all_consuming(identifier))))("-not-identifier")
                .unwrap_err()
        );
    }

    #[test]
    fn identifier_starts_with_number() {
        assert_eq!(
            parse_error(
                r#"0: at line 1, in identifier:
9identifier__
^"#
            ),
            interpret_error_for_test(cut(complete(all_consuming(identifier))))("9identifier__")
                .unwrap_err()
        );
    }

    #[test]
    fn parse_string_literal() {
        assert_eq!(
            "".to_string(),
            interpret_error_for_test(cut(complete(all_consuming(string_literal))))(r#""""#)
                .unwrap(),
        );
        assert_eq!(
            r#"""#.to_string(),
            interpret_error_for_test(cut(complete(all_consuming(string_literal))))(r#""\"""#)
                .unwrap(),
        );
        assert_eq!(
            r#"\"#.to_string(),
            interpret_error_for_test(cut(complete(all_consuming(string_literal))))(r#""\\""#)
                .unwrap(),
        );
        assert_eq!(
            r#""hello""world""#.to_string(),
            interpret_error_for_test(cut(complete(all_consuming(string_literal))))(
                r#""\"hello\"\"world\"""#
            )
            .unwrap(),
        );
    }

    #[test]
    fn parse_number_literal() {
        assert_eq!(
            0 as f64,
            interpret_error_for_test(cut(complete(all_consuming(number_literal))))("0").unwrap(),
        );
        assert_eq!(
            i32::MIN as f64,
            interpret_error_for_test(cut(complete(all_consuming(number_literal))))("-2147483648")
                .unwrap(),
        );
        assert_eq!(
            i32::MAX as f64,
            interpret_error_for_test(cut(complete(all_consuming(number_literal))))("2147483647")
                .unwrap(),
        );
        assert_eq!(
            u32::MAX as f64,
            interpret_error_for_test(cut(complete(all_consuming(number_literal))))("4294967295")
                .unwrap(),
        );
        assert_eq!(
            i64::MIN as f64,
            interpret_error_for_test(cut(complete(all_consuming(number_literal))))(
                "-9223372036854775808"
            )
            .unwrap(),
        );
        assert_eq!(
            i64::MAX as f64,
            interpret_error_for_test(cut(complete(all_consuming(number_literal))))(
                "9223372036854775807"
            )
            .unwrap(),
        );
        assert_eq!(
            u64::MAX as f64,
            interpret_error_for_test(cut(complete(all_consuming(number_literal))))(
                "18446744073709551615"
            )
            .unwrap(),
        );
    }

    #[test]
    fn field_string_enum() {
        let f = Field::StringEnum {
            name: "category".to_string(),
            values: vec![
                "ai".to_string(),
                "distributed systems".to_string(),
                "other".to_string(),
            ],
            default: Some("other".to_string()),
            on_conflict: OnConflict::Agreement,
        };
        let display = f.to_string();
        println!("{display}");
        assert_eq!(
            f,
            interpret_error_for_test(cut(complete(all_consuming(field))))(&display).unwrap(),
        );
    }

    #[test]
    fn readme_and_more() {
        let this = PolicyType {
            name: "policyai::EmailPolicy".to_string(),
            fields: vec![
                Field::Bool {
                    name: "unread".to_string(),
                    default: true,
                    on_conflict: OnConflict::Default,
                },
                Field::String {
                    name: "template".to_string(),
                    default: None,
                    on_conflict: OnConflict::Agreement,
                },
                Field::StringEnum {
                    name: "priority".to_string(),
                    values: vec!["low".to_string(), "medium".to_string(), "high".to_string()],
                    default: None,
                    on_conflict: OnConflict::LargestValue,
                },
                Field::StringEnum {
                    name: "category".to_string(),
                    values: vec![
                        "ai".to_string(),
                        "distributed systems".to_string(),
                        "other".to_string(),
                    ],
                    default: Some("other".to_string()),
                    on_conflict: OnConflict::Agreement,
                },
                Field::StringArray {
                    name: "labels".to_string(),
                },
                Field::Number {
                    name: "number".to_string(),
                    default: None,
                    on_conflict: OnConflict::Default,
                },
            ],
        };
        let display = this.to_string();
        println!("{display}");
        assert_eq!(
            this,
            interpret_error_for_test(cut(complete(all_consuming(policy_type))))(&display).unwrap(),
        );
    }
}
