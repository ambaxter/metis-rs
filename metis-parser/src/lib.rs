extern crate nom;
#[macro_use]
extern crate nom_locate;

use std::str::FromStr;

use nom::{FindSubstring, InputLength, InputTake, ParseTo};
use nom::branch::alt;
use nom::bytes::complete::{escaped, is_a, is_not, tag, take_till1, take_while1};
use nom::character::complete::{alphanumeric0, alphanumeric1, char, digit0, digit1, multispace0, multispace1, none_of, one_of};
use nom::character::is_digit;
use nom::combinator::{map, map_res, not, opt, recognize};
use nom::error::ErrorKind;
use nom::multi::{fold_many0, fold_many1};
use nom::sequence::{delimited, pair, preceded, separated_pair, terminated, tuple};
use nom_locate::{LocatedSpan, position};
use ordered_float::NotNan;

type Span<'a> = LocatedSpan<&'a str>;

fn span(i: &str) -> Span {
  Span::new(i)
}

#[derive(Copy, Clone, Debug)]
struct Spanned<'a, T> {
  s: Span<'a>,
  t: T,
}

fn stag<'a>(i: &'a str) -> impl Fn(Span<'a>) -> nom::IResult<Span<'a>, Span<'a>> {
  tag(span(i))
}

impl<'a, T> PartialEq for Spanned<'a, T>
  where T: PartialEq {
  fn eq(&self, other: &Self) -> bool {
    self.t == other.t &&
      self.s.fragment() == other.s.fragment()
  }
}

impl<'a, T> Eq for Spanned<'a, T>
  where T: Eq {}

impl<'a, T> AsRef<T> for Spanned<'a, T> {
  fn as_ref(&self) -> &T {
    &self.t
  }
}

fn pre_ms0<'a, I, O, E: nom::error::ParseError<I>, F>(f: F) -> impl Fn(I) -> nom::IResult<I, O, E>
  where F: Fn(I) -> nom::IResult<I, O, E>,
        I: nom::InputTakeAtPosition + nom::InputIter + nom::InputTake,
        <I as nom::InputTakeAtPosition>::Item: nom::AsChar + Clone,
{
  preceded(multispace0, f)
}

fn pre_ms1<'a, I, O, E: nom::error::ParseError<I>, F>(f: F) -> impl Fn(I) -> nom::IResult<I, O, E>
  where F: Fn(I) -> nom::IResult<I, O, E>,
        I: nom::InputTakeAtPosition + nom::InputIter + nom::InputTake,
        <I as nom::InputTakeAtPosition>::Item: nom::AsChar + Clone,
{
  preceded(multispace1, f)
}


#[derive(Default)]
pub struct ParseError;

impl std::fmt::Display for ParseError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "A parsing error occurred.")
  }
}

impl std::fmt::Debug for ParseError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    <ParseError as std::fmt::Display>::fmt(self, f)
  }
}

impl std::error::Error for ParseError {}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct Bool {
  val: bool
}

fn span_bool(s: Span) -> nom::IResult<Span, Span> {
  alt((
    stag("true"),
    stag("false")
  ))(s)
}

fn accept_bool(s: Span) -> nom::IResult<Span, Span> {
  pre_ms0(span_bool)(s)
}

fn parse_bool(s: Span) -> nom::IResult<Span, Spanned<Bool>> {
  map_res(
    accept_bool,
    |s| s.parse_to_result().map(|val| Spanned { s, t: Bool { val } }),
  )(s)
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Number {
  Integer(i64),
  Real(NotNan<f64>),
}

fn span_number(s: Span) -> nom::IResult<Span, Span> {
  recognize(tuple(
    (
      opt(stag("-")),
      opt(pair(digit0, stag("."))),
      digit1,
      opt(tuple(
        (
          one_of("eE"),
          opt(stag("-")),
          digit1
        )
      ))
    )
  ))(s)
}

fn accept_number(s: Span) -> nom::IResult<Span, Span> {
  pre_ms0(span_number)(s)
}

fn parse_to_number(s: Span) -> Result<Spanned<Number>, ErrorKind> {
  if !s.find_substring(".").is_some() &&
    !s.find_substring("e").is_some() &&
    !s.find_substring("E").is_some() {
    return s.parse_to_result()
      .map(|v| Spanned { s, t: Number::Integer(v) });
  }
  s.parse_to_result()
    .map(|v: f64| Spanned { s, t: Number::Real(v.into()) })
}

fn parse_number(s: Span) -> nom::IResult<Span, Spanned<Number>> {
  map_res(accept_number,
          parse_to_number,
  )(s)
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct QString<'a> {
  val: &'a str,
}

fn span_escaped_quotes(s: Span) -> nom::IResult<Span, Span> {
  let mut chars = s.fragment().char_indices();
  let mut prev = None;
  let mut this = chars.next();
  while let Some((n, c)) = this {
    if c == '"' {
      if let Some((_, p)) = prev {
        if p != '\\' {
          return Ok(s.take_split(n)).map(|(l, r)| (l, r));
        }
      } else {
        return Ok(s.take_split(n)).map(|(l, r)| (l, r));
      }
    }
    prev = this;
    this = chars.next();
  }
  Ok(s.take_split(s.input_len())).map(|(l, r)| (l, r))
}

fn accept_qstring(s: Span) -> nom::IResult<Span, Span> {
  pre_ms0(
    delimited(stag("\""), span_escaped_quotes, stag("\""))
  )(s)
}

fn parse_qstring(s: Span) -> nom::IResult<Span, Spanned<QString>> {
  map(accept_qstring,
      |s| Spanned { s, t: QString { val: s.fragment() } })
    (s)
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct Ident<'a> {
  val: &'a str
}

fn span_ident(s: Span) -> nom::IResult<Span, Span> {
  let fragment = s.fragment();
  if fragment.len() > 0 && !is_digit(fragment.bytes().nth(0).unwrap()) {
    return is_not(" \r\n\t\\'\"!@#$%^&*()-=+{}[];:<>,./?")(s);
  }
  Err(nom::Err::Error((s, ErrorKind::IsNot)))
}

fn accept_ident(s: Span) -> nom::IResult<Span, Span> {
  pre_ms0(span_ident)(s)
}

fn parse_ident(s: Span) -> nom::IResult<Span, Spanned<Ident>> {
  map(accept_ident,
      |s| Spanned { s, t: Ident { val: s.fragment() } })
    (s)
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct BIdent<'a> {
  val: &'a str
}

fn accept_bident(s: Span) -> nom::IResult<Span, Span> {
  pre_ms0(
    recognize(
      pair(
        opt(stag("$")),
        span_ident,
      )
    )
  )(s)
}

fn parse_bident(s: Span) -> nom::IResult<Span, Spanned<BIdent>> {
  map(accept_bident,
      |s| Spanned { s, t: BIdent { val: s.fragment() } },
  )(s)
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum AssignOp {
  AssignEq,
  PlusEq,
  MinusEq,
  MulEq,
  DivEq,
  AndEq,
  OrEq,
  XorEq,
}

impl FromStr for AssignOp {
  type Err = ErrorKind;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s {
      "=" => Ok(AssignOp::AssignEq),
      "+=" => Ok(AssignOp::PlusEq),
      "-=" => Ok(AssignOp::MinusEq),
      "*=" => Ok(AssignOp::MulEq),
      "/=" => Ok(AssignOp::DivEq),
      "&=" => Ok(AssignOp::DivEq),
      "|=" => Ok(AssignOp::OrEq),
      "^=" => Ok(AssignOp::XorEq),
      _ => Err(ErrorKind::ParseTo)
    }
  }
}


#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum UnaryOp {
  Not
}

impl FromStr for UnaryOp {
  type Err = ErrorKind;
  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s {
      "!" => Ok(UnaryOp::Not),
      _ => Err(ErrorKind::ParseTo)
    }
  }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum RelOp {
  Eq,
  Ne,
  Lt,
  Lte,
  Gt,
  Gte,
  Xor,
  And,
  Or,
}

impl FromStr for RelOp {
  type Err = ErrorKind;
  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s {
      "==" => Ok(RelOp::Eq),
      "!=" => Ok(RelOp::Ne),
      "<" => Ok(RelOp::Lt),
      "<=" => Ok(RelOp::Lte),
      ">" => Ok(RelOp::Gt),
      ">=" => Ok(RelOp::Gte),
      "^" => Ok(RelOp::Xor),
      "&&" => Ok(RelOp::And),
      "||" => Ok(RelOp::Or),
      _ => Err(ErrorKind::ParseTo)
    }
  }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum ArithOp {
  Mul,
  Div,
  Add,
  Sub,
}

impl FromStr for ArithOp {
  type Err = ErrorKind;
  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s {
      "*" => Ok(ArithOp::Mul),
      "/" => Ok(ArithOp::Div),
      "+" => Ok(ArithOp::Add),
      "-" => Ok(ArithOp::Sub),
      _ => Err(ErrorKind::ParseTo)
    }
  }
}

trait ParseToResult<R> {
  fn parse_to_result(&self) -> Result<R, ErrorKind>;
}

impl<T, R> ParseToResult<R> for T
  where T: ParseTo<R> {
  fn parse_to_result(&self) -> Result<R, ErrorKind> {
    self.parse_to().ok_or(ErrorKind::ParseTo)
  }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct MAccess<'a> {
  binding: Spanned<'a, BIdent<'a>>,
  member: Spanned<'a, Ident<'a>>,
}

fn parse_maccess(s: Span) -> nom::IResult<Span, MAccess> {
  map(
    pair(parse_bident, preceded(pre_ms0(stag(".")), parse_ident)),
    |(binding, member)| MAccess { binding, member },
  )(s)
}


#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum Expression<'a> {
  Bool(Spanned<'a, Bool>),
  Number(Spanned<'a, Number>),
  BIdent(Spanned<'a, BIdent<'a>>),
  QString(Spanned<'a, QString<'a>>),
  MAccess(MAccess<'a>),
  Grouping(Box<Expression<'a>>),
  Unary { op: Spanned<'a, UnaryOp>, right: Box<Expression<'a>> },
  ArithEq { op: Spanned<'a, ArithOp>, left: Box<Expression<'a>>, right: Box<Expression<'a>> },
  RelEq { op: Spanned<'a, RelOp>, left: Box<Expression<'a>>, right: Box<Expression<'a>> },
}

impl<'a> From<Spanned<'a, Bool>> for Expression<'a> {
  fn from(i: Spanned<'a, Bool>) -> Self {
    Expression::Bool(i)
  }
}

impl<'a> From<Spanned<'a, Number>> for Expression<'a> {
  fn from(i: Spanned<'a, Number>) -> Self {
    Expression::Number(i)
  }
}

impl<'a> From<Spanned<'a, BIdent<'a>>> for Expression<'a> {
  fn from(i: Spanned<'a, BIdent<'a>>) -> Self {
    Expression::BIdent(i)
  }
}

impl<'a> From<Spanned<'a, QString<'a>>> for Expression<'a> {
  fn from(i: Spanned<'a, QString<'a>>) -> Self {
    Expression::QString(i)
  }
}

impl<'a> From<MAccess<'a>> for Expression<'a> {
  fn from(i: MAccess<'a>) -> Self {
    Expression::MAccess(i)
  }
}

pub(crate) fn expression(s: Span) -> nom::IResult<Span, Expression> {
  logical(s)
}

pub(crate) fn logical(s: Span) -> nom::IResult<Span, Expression> {
  let parse_op = map_res(
    pre_ms0(alt((stag("&&"), stag("||"), stag("^")))),
    |s: Span| ParseToResult::<RelOp>::parse_to_result(&s)
      .map(|t| Spanned { s, t }),
  );
  let (s, left) = equality(s)?;
  fold_many0(
    pair(parse_op, equality),
    left,
    |left, (op, right)| Expression::RelEq { op, left: left.into(), right: right.into() },
  )(s)
}

pub(crate) fn equality(s: Span) -> nom::IResult<Span, Expression> {
  let parse_op = map_res(
    pre_ms0(alt((stag("=="), stag("!=")))),
    |s: Span| ParseToResult::<RelOp>::parse_to_result(&s)
      .map(|t| Spanned { s, t }),
  );
  let (s, left) = comparison(s)?;
  fold_many0(
    pair(parse_op, comparison),
    left,
    |left, (op, right)| Expression::RelEq { op, left: left.into(), right: right.into() },
  )(s)
}

pub(crate) fn comparison(s: Span) -> nom::IResult<Span, Expression> {
  let parse_op = map_res(
    pre_ms0(alt((stag("<="), stag("<"), stag(">="), stag(">")))),
    |s: Span| ParseToResult::<RelOp>::parse_to_result(&s)
      .map(|t| Spanned { s, t }),
  );
  let (s, left) = addition(s)?;
  fold_many0(
    pair(parse_op, addition),
    left,
    |left, (op, right)| Expression::RelEq { op, left: left.into(), right: right.into() },
  )(s)
}

pub(crate) fn addition(s: Span) -> nom::IResult<Span, Expression> {
  let parse_op = map_res(
    pre_ms0(alt((stag("+"), stag("-")))),
    |s: Span| ParseToResult::<ArithOp>::parse_to_result(&s)
      .map(|t| Spanned { s, t }),
  );
  let (s, left) = multiplication(s)?;
  fold_many0(
    pair(parse_op, multiplication),
    left,
    |left, (op, right)| Expression::ArithEq { op, left: left.into(), right: right.into() },
  )(s)
}

pub(crate) fn multiplication(s: Span) -> nom::IResult<Span, Expression> {
  let parse_op = map_res(
    pre_ms0(alt((stag("*"), stag("/")))),
    |s: Span| ParseToResult::<ArithOp>::parse_to_result(&s)
      .map(|t| Spanned { s, t }),
  );
  let (s, left) = unary(s)?;
  fold_many0(
    pair(parse_op, unary),
    left,
    |left, (op, right)| Expression::ArithEq { op, left: left.into(), right: right.into() },
  )(s)
}

pub(crate) fn unary(s: Span) -> nom::IResult<Span, Expression> {
  let parse_unary = pair(
    map_res(
      pre_ms0(stag("!")),
      |s: Span| ParseToResult::<UnaryOp>::parse_to_result(&s)
        .map(|t| Spanned { s, t }),
    ),
    unary,
  );

  alt((
    map(parse_unary,
        |(op, exp)| Expression::Unary { op, right: exp.into() },
    ),
    primary
  ))(s)
}


pub(crate) fn primary(s: Span) -> nom::IResult<Span, Expression> {
  alt((
    map(parse_bool, |i| Into::<Expression>::into(i)),
    map(parse_qstring, |i| Into::<Expression>::into(i)),
    map(parse_number, |i| Into::<Expression>::into(i)),
    map(parse_maccess, |i| Into::<Expression>::into(i)),
    map(parse_bident, |i| Into::<Expression>::into(i)),
    map(
      delimited(pre_ms0(stag("(")), expression, pre_ms0(stag(")"))),
      |e| Expression::Grouping(e.into()),
    )
  ))(s)
}

pub(crate) mod lhs {
  use super::*;

  #[derive(Clone, PartialEq, Eq, Debug)]
  pub(crate) struct Constraint<'a> {
    b: Option<Spanned<'a, BIdent<'a>>>,
    exp: Expression<'a>,
  }

  pub(crate) fn parse_constraint(i: Span) -> nom::IResult<Span, Constraint> {
    map(
      pair(
        opt(terminated(parse_bident, pre_ms0(stag(":")))),
        expression,
      ),
      |(b, exp)| Constraint { b, exp },
    )(i)
  }

  #[derive(Clone, PartialEq, Eq, Debug)]
  pub(crate) struct Condition<'a> {
    b: Option<Spanned<'a, BIdent<'a>>>,
    t: Spanned<'a, Ident<'a>>,
    con: Vec<Constraint<'a>>,
  }

  pub(crate) fn parse_condition(s: Span) -> nom::IResult<Span, Condition> {
    let condition_ident = opt(terminated(parse_bident, pre_ms0(stag(":"))));
    let constraints = fold_many0(
      terminated(parse_constraint, opt(pre_ms0(stag(",")))),
      Vec::with_capacity(0),
      |mut v, constraint| {
        v.push(constraint);
        v
      },
    );
    map(tuple((
      condition_ident,
      parse_ident,
      delimited(pre_ms0(stag("(")), constraints, pre_ms0(stag(")")))
    )),
        |(b, t, con)| Condition { b, t, con },
    )(s)
  }

  #[derive(Clone, PartialEq, Eq, Debug)]
  pub(crate) struct Eval<'a> {
    exp: Expression<'a>
  }

  pub(crate) fn parse_eval(s: Span) -> nom::IResult<Span, Eval> {
    map(
      preceded(pre_ms0(stag("eval")), delimited(pre_ms0(stag("(")), unary, pre_ms0(stag(")")))),
      |exp| Eval { exp },
    )(s)
  }

  #[derive(Clone, PartialEq, Eq, Debug)]
  pub(crate) enum Statement<'a> {
    Condition(Condition<'a>),
    Eval(Eval<'a>),
  }

  pub(crate) fn parse_statement(s: Span) -> nom::IResult<Span, Statement> {
    alt((
      map(parse_eval, |eval| Statement::Eval(eval)),
      map(parse_condition, |condition| Statement::Condition(condition))
    ))(s)
  }

  #[derive(Clone, PartialEq, Eq, Debug)]
  pub(crate) struct LHS<'a> {
    lhs: Vec<Statement<'a>>
  }

  pub(crate) fn parse_lhs(s: Span) -> nom::IResult<Span, LHS> {
    let statements = fold_many0(
      parse_statement,
      Vec::with_capacity(0),
      |mut v, statement| {
        v.push(statement);
        v
      },
    );
    map(statements, |lhs| LHS { lhs })(s)
  }
}

pub(crate) mod rhs {
  use super::*;

  #[derive(Clone, PartialEq, Eq, Debug)]
  pub(crate) enum Assignable<'a> {
    BIdent(Spanned<'a, BIdent<'a>>),
    MAccess(MAccess<'a>),
  }

  pub(crate) fn parse_assignable(s: Span) -> nom::IResult<Span, Assignable> {
    alt((
      map(parse_maccess, |m| Assignable::MAccess(m)),
      map(parse_bident, |b| Assignable::BIdent(b))
    ))(s)
  }

  #[derive(Clone, PartialEq, Eq, Debug)]
  pub(crate) enum BindingType<'a> {
    Default,
    Clone(Spanned<'a, BIdent<'a>>),
    Exp(Expression<'a>)
  }

  #[derive(Clone, PartialEq, Eq, Debug)]
  pub(crate) enum Statement<'a> {
    Binding { left: Spanned<'a, BIdent<'a>>, t: Option<Spanned<'a, Ident<'a>>>, right: BindingType<'a> },
    Assignment { op: Spanned<'a, AssignOp>, left: Assignable<'a>, right: Expression<'a> },
    Update(Spanned<'a, BIdent<'a>>),
    Insert(Spanned<'a, BIdent<'a>>),
    Retract(Spanned<'a, BIdent<'a>>),
  }

  fn par_wrapped_bident(i: Span) -> nom::IResult<Span, Spanned<BIdent>> {
    delimited(pre_ms0(stag("(")), parse_bident, pre_ms0(stag(")")))(i)
  }


  fn empty_par(i: Span) -> nom::IResult<Span, Span> {
    delimited(pre_ms0(stag("(")), multispace0, pre_ms0(stag(")")))(i)
  }

  pub(crate) fn parse_statement(s: Span) -> nom::IResult<Span, Statement> {

    let map_update = map(
      preceded(pre_ms0(stag("update")),
               par_wrapped_bident),
      |b| Statement::Update(b),
    );
    let map_insert = map(
      preceded(pre_ms0(stag("insert")),
               par_wrapped_bident),
      |b| Statement::Insert(b),
    );
    let map_retract = map(
      preceded(pre_ms0(stag("retract")),
               par_wrapped_bident),
      |b| Statement::Retract(b),
    );
    let map_default_or_exp = alt((
      map(preceded(pre_ms0(stag("clone")), par_wrapped_bident),
          |b| BindingType::Clone(b)),
      map(preceded(pre_ms0(stag("default")), empty_par),
          |_| BindingType::Default),
      map(expression, |e| BindingType::Exp(e))
    ));

    let map_binding = map(
      tuple((
        preceded(pre_ms0(stag("let")), parse_bident),
        opt(preceded(pre_ms0(stag(":")), pre_ms0(parse_ident))),
        preceded(pre_ms0(stag("=")), map_default_or_exp))),
      |(left, t, right)| Statement::Binding { left, t, right },
    );

    let parse_assign_op = map_res(
      pre_ms0(alt((stag("="), stag("+="), stag("-="), stag("*="), stag("/="), stag("&="), stag("|="), stag("^=")))),
      |s: Span| ParseToResult::<AssignOp>::parse_to_result(&s)
        .map(|t| Spanned { s, t }),
    );
    let map_assignment = map(
      tuple((
        parse_assignable, parse_assign_op, expression
      )),
      |(left, op, right)| Statement::Assignment { left, op, right },
    );

    terminated(
      alt((
        map_update,
        map_insert,
        map_retract,
        map_binding,
        map_assignment
      )),
      pre_ms0(stag(";")),
    )(s)
  }

  #[derive(Clone, PartialEq, Eq, Debug)]
  pub(crate) struct RHS<'a> {
    rhs: Vec<Statement<'a>>
  }

  pub(crate) fn parse_rhs(s: Span) -> nom::IResult<Span, RHS> {
    let statements = fold_many0(
      parse_statement,
      Vec::with_capacity(0),
      |mut v, statement| {
        v.push(statement);
        v
      },
    );
    map(statements, |rhs| RHS { rhs })(s)
  }
}

/*

#[derive(Clone, PartialEq, Eq, Default, Debug)]
pub struct Rule<'a> {
  pub name: &'a str,
}

// Rule name (doesn't start with a number or whitespace. quoted may contain whitespace. may not contain quotes
// when Option<condition_bind:> Condition(Option<constraint>*)*
fn rule_parser(i: &str) -> nom::IResult<&str, Rule> {
  let (i, _) = tag("rule")(i)?;
  let (i, _) = multispace1(i)?;
  let (i, name) = rule_name(i)?;
  let (i, _) = multispace1(i)?;
  let (i, _) = parse_when(i)?;
  let (i, _) = multispace1(i)?;
  let (i, _) = then(i)?;
  let (i, _) = multispace1(i)?;
  let (i, _) = end(i)?;
  Ok(
    (i,
      Rule{ name }
    )
  )
}
fn rule_name(i: &str) -> nom::IResult<&str, &str> {
  alt(
    (
      delimited(tag("\""),
                accept_escaped_quotes,
                tag("\"")),
      accept_ident
    )
  )(i)
}


#[derive(Clone, PartialEq, Eq, Debug)]
enum AssignDest<'a> {
  BIdent(BIdent<'a>),
  MAccess(MAccess<'a>)
}

impl<'a> From<BIdent<'a>> for AssignDest<'a> {
  fn from(i: BIdent<'a>) -> Self {
    AssignDest::BIdent(i)
  }
}

impl<'a> From<MAccess<'a>> for AssignDest<'a> {
  fn from(i: MAccess<'a>) -> Self {
    AssignDest::MAccess(i)
  }
}


#[derive(Clone, PartialEq, Eq, Debug)]
enum Statement<'a> {
  Assign{op: AssignOp, left: AssignDest<'a>, right: Box<Expression<'a>>}
}

fn parse_when(i: &str) -> nom::IResult<&str, &str> {
  let (i, when) = opt(tag("when"))(i)?;
  if let Some(_) = when {
    let (i, _) = multispace1(i)?;
    let (i, ident) = ident(i)?;
    let (i, del) = delimited(tag("("), alphanumeric0, tag(")"))(i)?;
    Ok((i, &""))
  } else {
    Ok((i, &""))
  }
}

fn then(i: &str)  -> nom::IResult<&str, &str> {
  tag("then")(i)
}

fn end(i: &str)  -> nom::IResult<&str, &str> {
  tag("end")(i)
}
*/

#[cfg(test)]
mod tests {
  use nom::bytes::complete::take_while;
  use nom::character::complete::anychar;
  use nom::Err::Error;
  use nom::error::ErrorKind;
  use nom::IResult;

  use super::*;

  #[test]
  fn test_bool() {
    println!("{:?}", parse_bool(Span::new("true  ")));
  }

  #[test]
  fn test_number() {
    println!("{:?}", parse_number(Span::new("-1")));
    println!("{:?}", parse_number(Span::new("9223372036854775808")));
  }

  #[test]
  fn test_accept_quotes() {
    println!("{:?}", lhs::parse_condition(span("ABC1(123 != \"me\")")));
  }

  #[test]
  fn test_rhs() {
    println!("{:?}", rhs::parse_rhs(span("let b: New = default(); let c = clone($bd); insert(b); insert($bd); ")));
  }

  /*
    #[test]
    fn test_empty_rule() {
      let empty_rule_mrl= "rule \"empty\"\n\
      then\n\
      end";
      println!("{:?}", rule_parser(empty_rule_mrl));
    }


    #[test]
    fn test_simple_when() {
      let simple_when_mrl = "rule \"simple_when\"\n\
      when\n\
      AType()
      then\n\
      end";
      println!("{:?}", rule_parser(simple_when_mrl));
    }

    #[test]
    fn test_number() {
      assert_eq!(Ok(("", Number{s: "0"})), number("0"));
      assert_eq!(Ok(("", Number{s: "0.0"})), number("0.0"));
      assert_eq!(Ok(("", Number{s: "-3"})), number("-3"));
      assert_eq!(Ok(("", Number{s: "-3.0"})), number("-3.0"));
      assert_eq!(Ok(("", Number{s: "37.0e37"})), number("37.0e37"));
      assert_eq!(Ok(("", Number{s: "37.0e-37"})), number("37.0e-37"));
      assert_eq!(Err(Error(("x0x", ErrorKind::Digit))), number("x0x"));
    }

    #[test]
    fn test_quoted_string() {
      assert_eq!(Ok(("abc123", QString {s: "test"})), quoted_string("\"test\"abc123"));
      assert_eq!(Ok(("abc123", QString {s: "test\\\""})), quoted_string("\"test\\\"\"abc123"));
      assert_eq!(Ok(("abc123", QString {s: "test"})), quoted_string("\"test\"abc123"));
    }

    #[test]
    fn test_ident() {
      assert_eq!(Ok((".abc(1234)", Ident{s: "ABC_123"})),  ident("ABC_123.abc(1234)"));
      assert_eq!(Ok(("", Ident{s: "ABC_123"})),  ident("ABC_123"));
      assert_eq!(Err(Error(("1ABC", ErrorKind::IsNot))), ident("1ABC"));
    }

    #[test]
    fn test_bound_ident() {
      assert_eq!(Ok((".abc(1234)", BIdent{s: "ABC_123"})),  bindable_ident("ABC_123.abc(1234)"));
      assert_eq!(Ok(("", BIdent{s: "ABC_123"})),  bindable_ident("ABC_123"));
      assert_eq!(Err(Error(("1ABC", ErrorKind::IsNot))), bindable_ident("1ABC"));
      assert_eq!(Ok((".abc(1234)", BIdent {s: "$ABC_123"})), bindable_ident("$ABC_123.abc(1234)"));
      assert_eq!(Ok(("", BIdent {s: "$ABC_123"})), bindable_ident("$ABC_123"));
      assert_eq!(Err(Error(("1ABC", ErrorKind::IsNot))), bindable_ident("$1ABC"));
    }

    #[test]
    fn test_expression() {
      println!("{:?}", bound_condition("$a7: Condition($a: 83 + 6, a > 6)"));
    }*/
}
