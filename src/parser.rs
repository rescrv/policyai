use std::fmt;

use crate::{t64, Field, OnConflict, PolicyType};

#[derive(Debug, Clone, PartialEq)]
pub struct Position {
    pub line: usize,
    pub column: usize,
}

impl Position {
    fn new(line: usize, column: usize) -> Self {
        Self { line, column }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    UnexpectedToken {
        expected: String,
        found: String,
        position: Position,
    },
    UnexpectedEndOfInput {
        expected: String,
        position: Position,
    },
    InvalidIdentifier {
        reason: String,
        position: Position,
    },
    InvalidStringLiteral {
        reason: String,
        position: Position,
    },
    InvalidNumber {
        reason: String,
        position: Position,
    },
    DuplicateFieldName {
        name: String,
        position: Position,
    },
    Custom {
        message: String,
        position: Position,
    },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::UnexpectedToken {
                expected,
                found,
                position,
            } => {
                write!(
                    f,
                    "at line {}:{}: expected {expected}, found '{found}'",
                    position.line, position.column
                )
            }
            ParseError::UnexpectedEndOfInput { expected, position } => {
                write!(
                    f,
                    "at line {}:{}: unexpected end of input, expected {expected}",
                    position.line, position.column
                )
            }
            ParseError::InvalidIdentifier { reason, position } => {
                write!(
                    f,
                    "at line {}:{}: invalid identifier: {reason}",
                    position.line, position.column
                )
            }
            ParseError::InvalidStringLiteral { reason, position } => {
                write!(
                    f,
                    "at line {}:{}: invalid string literal: {reason}",
                    position.line, position.column
                )
            }
            ParseError::InvalidNumber { reason, position } => {
                write!(
                    f,
                    "at line {}:{}: invalid number: {reason}",
                    position.line, position.column
                )
            }
            ParseError::DuplicateFieldName { name, position } => {
                write!(
                    f,
                    "at line {}:{}: duplicate field name '{name}'",
                    position.line, position.column
                )
            }
            ParseError::Custom { message, position } => {
                write!(
                    f,
                    "at line {}:{}: {message}",
                    position.line, position.column
                )
            }
        }
    }
}

impl std::error::Error for ParseError {}

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    Type,
    Bool,
    String,
    Number,
    True,
    False,

    // Identifiers and literals
    Identifier(String),
    StringLiteral(String),
    NumberLiteral(f64),

    // Symbols
    LeftBrace,
    RightBrace,
    LeftBracket,
    RightBracket,
    Colon,
    Comma,
    Equals,
    At,
    DoubleColon,

    // Special conflict resolution keywords
    Agreement,
    Sticky,
    Wins,
    Last,
    Highest,
    Largest,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Type => write!(f, "type"),
            Token::Bool => write!(f, "bool"),
            Token::String => write!(f, "string"),
            Token::Number => write!(f, "number"),
            Token::True => write!(f, "true"),
            Token::False => write!(f, "false"),
            Token::Identifier(s) => write!(f, "{s}"),
            Token::StringLiteral(s) => write!(f, "\"{s}\""),
            Token::NumberLiteral(n) => write!(f, "{n}"),
            Token::LeftBrace => write!(f, "{{"),
            Token::RightBrace => write!(f, "}}"),
            Token::LeftBracket => write!(f, "["),
            Token::RightBracket => write!(f, "]"),
            Token::Colon => write!(f, ":"),
            Token::Comma => write!(f, ","),
            Token::Equals => write!(f, "="),
            Token::At => write!(f, "@"),
            Token::DoubleColon => write!(f, "::"),
            Token::Agreement => write!(f, "agreement"),
            Token::Sticky => write!(f, "sticky"),
            Token::Wins => write!(f, "wins"),
            Token::Last => write!(f, "last"),
            Token::Highest => write!(f, "highest"),
            Token::Largest => write!(f, "largest"),
        }
    }
}

pub struct Lexer {
    input: Vec<char>,
    position: usize,
    line: usize,
    column: usize,
}

impl Lexer {
    pub fn new(input: &str) -> Self {
        Self {
            input: input.chars().collect(),
            position: 0,
            line: 1,
            column: 1,
        }
    }

    fn current_position(&self) -> Position {
        Position::new(self.line, self.column)
    }

    fn peek(&self) -> Option<char> {
        self.input.get(self.position).copied()
    }

    fn advance(&mut self) -> Option<char> {
        if let Some(ch) = self.peek() {
            self.position += 1;
            if ch == '\n' {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
            Some(ch)
        } else {
            None
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn read_identifier(&mut self) -> String {
        let mut ident = String::new();
        while let Some(ch) = self.peek() {
            if ch.is_alphanumeric() || ch == '_' {
                ident.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        ident
    }

    fn read_string_literal(&mut self) -> Result<String, ParseError> {
        let start_pos = self.current_position();

        // Skip opening quote
        self.advance();

        let mut result = String::new();
        let mut escaped = false;

        loop {
            match self.peek() {
                None => {
                    return Err(ParseError::InvalidStringLiteral {
                        reason: "unterminated string literal".to_string(),
                        position: start_pos,
                    });
                }
                Some('\\') if !escaped => {
                    escaped = true;
                    self.advance();
                }
                Some('"') if !escaped => {
                    self.advance();
                    return Ok(result);
                }
                Some(ch) => {
                    if escaped {
                        match ch {
                            '"' | '\\' => result.push(ch),
                            _ => {
                                return Err(ParseError::InvalidStringLiteral {
                                    reason: format!("invalid escape sequence '\\{ch}'"),
                                    position: self.current_position(),
                                });
                            }
                        }
                        escaped = false;
                    } else {
                        result.push(ch);
                    }
                    self.advance();
                }
            }
        }
    }

    fn read_number(&mut self) -> Result<f64, ParseError> {
        let start_pos = self.current_position();
        let mut num_str = String::new();

        // Handle negative sign
        if self.peek() == Some('-') {
            num_str.push('-');
            self.advance();
        }

        // Read digits and decimal point
        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() || (ch == '.' && !num_str.contains('.')) {
                num_str.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        num_str
            .parse::<f64>()
            .map_err(|_| ParseError::InvalidNumber {
                reason: format!("'{num_str}' is not a valid number"),
                position: start_pos,
            })
    }

    pub fn tokenize(&mut self) -> Result<Vec<(Token, Position)>, ParseError> {
        let mut tokens = Vec::new();

        loop {
            self.skip_whitespace();

            let pos = self.current_position();

            match self.peek() {
                None => break,
                Some('"') => {
                    let string_lit = self.read_string_literal()?;
                    tokens.push((Token::StringLiteral(string_lit), pos));
                }
                Some('-') | Some('0'..='9') => {
                    let num = self.read_number()?;
                    tokens.push((Token::NumberLiteral(num), pos));
                }
                Some('{') => {
                    self.advance();
                    tokens.push((Token::LeftBrace, pos));
                }
                Some('}') => {
                    self.advance();
                    tokens.push((Token::RightBrace, pos));
                }
                Some('[') => {
                    self.advance();
                    tokens.push((Token::LeftBracket, pos));
                }
                Some(']') => {
                    self.advance();
                    tokens.push((Token::RightBracket, pos));
                }
                Some(':') => {
                    self.advance();
                    if self.peek() == Some(':') {
                        self.advance();
                        tokens.push((Token::DoubleColon, pos));
                    } else {
                        tokens.push((Token::Colon, pos));
                    }
                }
                Some(',') => {
                    self.advance();
                    tokens.push((Token::Comma, pos));
                }
                Some('=') => {
                    self.advance();
                    tokens.push((Token::Equals, pos));
                }
                Some('@') => {
                    self.advance();
                    tokens.push((Token::At, pos));
                }
                Some(ch) if ch.is_alphabetic() || ch == '_' => {
                    let ident = self.read_identifier();
                    let token = match ident.as_str() {
                        "type" => Token::Type,
                        "bool" => Token::Bool,
                        "string" => Token::String,
                        "number" => Token::Number,
                        "true" => Token::True,
                        "false" => Token::False,
                        "agreement" => Token::Agreement,
                        "sticky" => Token::Sticky,
                        "wins" => Token::Wins,
                        "last" => Token::Last,
                        "highest" => Token::Highest,
                        "largest" => Token::Largest,
                        _ => Token::Identifier(ident),
                    };
                    tokens.push((token, pos));
                }
                Some(ch) => {
                    return Err(ParseError::Custom {
                        message: format!("unexpected character '{ch}'"),
                        position: pos,
                    });
                }
            }
        }

        Ok(tokens)
    }
}

pub struct Parser {
    tokens: Vec<(Token, Position)>,
    position: usize,
}

impl Parser {
    pub fn new(tokens: Vec<(Token, Position)>) -> Self {
        Self {
            tokens,
            position: 0,
        }
    }

    fn current_position(&self) -> Position {
        self.tokens
            .get(self.position)
            .map(|(_, pos)| pos.clone())
            .unwrap_or_else(|| {
                self.tokens
                    .last()
                    .map(|(_, pos)| Position::new(pos.line, pos.column + 1))
                    .unwrap_or_else(|| Position::new(1, 1))
            })
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.position).map(|(token, _)| token)
    }

    fn advance(&mut self) -> Option<Token> {
        if self.position < self.tokens.len() {
            let token = self.tokens[self.position].0.clone();
            self.position += 1;
            Some(token)
        } else {
            None
        }
    }

    fn expect(&mut self, expected: Token) -> Result<(), ParseError> {
        let pos = self.current_position();
        match self.peek() {
            Some(token) if *token == expected => {
                self.advance();
                Ok(())
            }
            Some(token) => Err(ParseError::UnexpectedToken {
                expected: expected.to_string(),
                found: token.to_string(),
                position: pos,
            }),
            None => Err(ParseError::UnexpectedEndOfInput {
                expected: expected.to_string(),
                position: pos,
            }),
        }
    }

    fn parse_identifier(&mut self) -> Result<String, ParseError> {
        let pos = self.current_position();
        match self.advance() {
            Some(Token::Identifier(name)) => Ok(name),
            Some(token) => Err(ParseError::UnexpectedToken {
                expected: "identifier".to_string(),
                found: token.to_string(),
                position: pos,
            }),
            None => Err(ParseError::UnexpectedEndOfInput {
                expected: "identifier".to_string(),
                position: pos,
            }),
        }
    }

    fn parse_string_literal(&mut self) -> Result<String, ParseError> {
        let pos = self.current_position();
        match self.advance() {
            Some(Token::StringLiteral(s)) => Ok(s),
            Some(token) => Err(ParseError::UnexpectedToken {
                expected: "string literal".to_string(),
                found: token.to_string(),
                position: pos,
            }),
            None => Err(ParseError::UnexpectedEndOfInput {
                expected: "string literal".to_string(),
                position: pos,
            }),
        }
    }

    fn parse_number_literal(&mut self) -> Result<f64, ParseError> {
        let pos = self.current_position();
        match self.advance() {
            Some(Token::NumberLiteral(n)) => Ok(n),
            Some(token) => Err(ParseError::UnexpectedToken {
                expected: "number literal".to_string(),
                found: token.to_string(),
                position: pos,
            }),
            None => Err(ParseError::UnexpectedEndOfInput {
                expected: "number literal".to_string(),
                position: pos,
            }),
        }
    }

    fn parse_bool_conflict(&mut self) -> Result<OnConflict, ParseError> {
        if self.peek() == Some(&Token::At) {
            self.advance();
            match self.peek() {
                Some(Token::Sticky) => {
                    self.advance();
                    Ok(OnConflict::LargestValue)
                }
                Some(Token::Agreement) => {
                    self.advance();
                    Ok(OnConflict::Agreement)
                }
                _ => {
                    let pos = self.current_position();
                    Err(ParseError::Custom {
                        message: "expected 'sticky' or 'agreement' after '@'".to_string(),
                        position: pos,
                    })
                }
            }
        } else {
            Ok(OnConflict::Default)
        }
    }

    fn parse_string_conflict(&mut self) -> Result<OnConflict, ParseError> {
        if self.peek() == Some(&Token::At) {
            self.advance();
            if self.peek() == Some(&Token::Last) {
                self.advance();
                self.expect(Token::Wins)?;
                Ok(OnConflict::LargestValue)
            } else if self.peek() == Some(&Token::Agreement) {
                self.advance();
                Ok(OnConflict::Agreement)
            } else {
                let pos = self.current_position();
                Err(ParseError::Custom {
                    message: "expected 'last wins' or 'agreement' after '@'".to_string(),
                    position: pos,
                })
            }
        } else {
            Ok(OnConflict::Default)
        }
    }

    fn parse_string_enum_conflict(&mut self) -> Result<OnConflict, ParseError> {
        if self.peek() == Some(&Token::At) {
            self.advance();
            if self.peek() == Some(&Token::Highest) {
                self.advance();
                self.expect(Token::Wins)?;
                Ok(OnConflict::LargestValue)
            } else if self.peek() == Some(&Token::Agreement) {
                self.advance();
                Ok(OnConflict::Agreement)
            } else {
                let pos = self.current_position();
                Err(ParseError::Custom {
                    message: "expected 'highest wins' or 'agreement' after '@'".to_string(),
                    position: pos,
                })
            }
        } else {
            Ok(OnConflict::Default)
        }
    }

    fn parse_number_conflict(&mut self) -> Result<OnConflict, ParseError> {
        if self.peek() == Some(&Token::At) {
            self.advance();
            if matches!(self.peek(), Some(&Token::Last) | Some(&Token::Largest)) {
                self.advance();
                self.expect(Token::Wins)?;
                Ok(OnConflict::LargestValue)
            } else if self.peek() == Some(&Token::Agreement) {
                self.advance();
                Ok(OnConflict::Agreement)
            } else {
                let pos = self.current_position();
                Err(ParseError::Custom {
                    message: "expected 'last wins', 'largest wins', or 'agreement' after '@'"
                        .to_string(),
                    position: pos,
                })
            }
        } else {
            Ok(OnConflict::Default)
        }
    }

    fn parse_field(&mut self) -> Result<Field, ParseError> {
        let name = self.parse_identifier()?;
        self.expect(Token::Colon)?;

        match self.peek() {
            Some(Token::Bool) => {
                self.advance();
                let on_conflict = self.parse_bool_conflict()?;
                let default = if self.peek() == Some(&Token::Equals) {
                    self.advance();
                    match self.advance() {
                        Some(Token::True) => true,
                        Some(Token::False) => false,
                        _ => {
                            return Err(ParseError::Custom {
                                message: "expected 'true' or 'false' after '='".to_string(),
                                position: self.current_position(),
                            });
                        }
                    }
                } else {
                    false
                };
                Ok(Field::Bool {
                    name,
                    on_conflict,
                    default,
                })
            }
            Some(Token::String) => {
                self.advance();
                let on_conflict = self.parse_string_conflict()?;
                let default = if self.peek() == Some(&Token::Equals) {
                    self.advance();
                    Some(self.parse_string_literal()?)
                } else {
                    None
                };
                Ok(Field::String {
                    name,
                    on_conflict,
                    default,
                })
            }
            Some(Token::Number) => {
                self.advance();
                let on_conflict = self.parse_number_conflict()?;
                let default = if self.peek() == Some(&Token::Equals) {
                    self.advance();
                    Some(t64(self.parse_number_literal()?))
                } else {
                    None
                };
                Ok(Field::Number {
                    name,
                    on_conflict,
                    default,
                })
            }
            Some(Token::LeftBracket) => {
                self.advance();
                if self.peek() == Some(&Token::String) {
                    self.advance();
                    self.expect(Token::RightBracket)?;
                    Ok(Field::StringArray { name })
                } else {
                    // String enum
                    let mut values = vec![self.parse_string_literal()?];
                    while self.peek() == Some(&Token::Comma) {
                        self.advance();
                        values.push(self.parse_string_literal()?);
                    }
                    self.expect(Token::RightBracket)?;
                    let on_conflict = self.parse_string_enum_conflict()?;
                    let default = if self.peek() == Some(&Token::Equals) {
                        self.advance();
                        Some(self.parse_string_literal()?)
                    } else {
                        None
                    };
                    Ok(Field::StringEnum {
                        name,
                        values,
                        on_conflict,
                        default,
                    })
                }
            }
            _ => {
                let pos = self.current_position();
                Err(ParseError::Custom {
                    message: "expected field type (bool, string, number, or [...)".to_string(),
                    position: pos,
                })
            }
        }
    }

    pub fn parse_policy_type(&mut self) -> Result<PolicyType, ParseError> {
        self.expect(Token::Type)?;

        // Parse name (can be namespaced with ::)
        let mut name_parts = vec![self.parse_identifier()?];
        while self.peek() == Some(&Token::DoubleColon) {
            self.advance();
            name_parts.push(self.parse_identifier()?);
        }
        let name = name_parts.join("::");

        self.expect(Token::LeftBrace)?;

        let mut fields = Vec::new();
        let mut field_names = std::collections::HashSet::new();

        // Parse fields
        while self.peek() != Some(&Token::RightBrace) && self.peek().is_some() {
            let field = self.parse_field()?;

            // Check for duplicate field names
            let field_name = match &field {
                Field::Bool { name, .. }
                | Field::String { name, .. }
                | Field::StringEnum { name, .. }
                | Field::StringArray { name }
                | Field::Number { name, .. } => name.clone(),
            };

            if !field_names.insert(field_name.clone()) {
                return Err(ParseError::DuplicateFieldName {
                    name: field_name,
                    position: self.current_position(),
                });
            }

            fields.push(field);

            // Handle optional comma
            if self.peek() == Some(&Token::Comma) {
                self.advance();
            } else if self.peek() != Some(&Token::RightBrace) {
                return Err(ParseError::Custom {
                    message: "expected ',' or '}' after field definition".to_string(),
                    position: self.current_position(),
                });
            }
        }

        self.expect(Token::RightBrace)?;

        Ok(PolicyType { name, fields })
    }
}

pub fn parse(input: &str) -> Result<PolicyType, ParseError> {
    let mut lexer = Lexer::new(input);
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new(tokens);
    parser.parse_policy_type()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lexer_simple() {
        let mut lexer = Lexer::new("type Test { }");
        let tokens = lexer.tokenize().unwrap();
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].0, Token::Type);
        assert_eq!(tokens[1].0, Token::Identifier("Test".to_string()));
        assert_eq!(tokens[2].0, Token::LeftBrace);
        assert_eq!(tokens[3].0, Token::RightBrace);
    }

    #[test]
    fn test_parse_simple() {
        let result = parse("type Test { }");
        assert!(result.is_ok());
        let policy_type = result.unwrap();
        assert_eq!(policy_type.name, "Test");
        assert_eq!(policy_type.fields.len(), 0);
    }

    #[test]
    fn test_parse_bool_field() {
        let result = parse("type Test { active: bool = true }");
        assert!(result.is_ok());
        let policy_type = result.unwrap();
        assert_eq!(policy_type.fields.len(), 1);
        match &policy_type.fields[0] {
            Field::Bool { name, default, .. } => {
                assert_eq!(name, "active");
                assert!(*default);
            }
            _ => panic!("Expected bool field"),
        }
    }

    #[test]
    fn test_parse_data_policy_file() {
        const POLICY_CONTENT: &str = include_str!("../data/policy");
        let result = parse(POLICY_CONTENT);

        match &result {
            Err(e) => panic!("Failed to parse data/policy: {e}"),
            Ok(policy_type) => {
                assert_eq!(policy_type.name, "policyai::EmailPolicy");
                assert_eq!(policy_type.fields.len(), 6);
            }
        }
    }
}
