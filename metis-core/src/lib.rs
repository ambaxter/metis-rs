extern crate float_cmp;
extern crate num_traits;
extern crate ordered_float;

use std::any::{Any, TypeId};

use float_cmp::ApproxEq;
use num_traits::{Float, Unsigned, Zero, One};
use ordered_float::NotNan;
use std::path::Iter;
use std::iter::FusedIterator;
use std::hash::Hash;
use std::marker::PhantomData;
use string_interner::Symbol;
use std::fmt::Debug;
use dyn_clone::DynClone;

pub trait StaticFactHashEq {
  type Item: Hash + Eq + Debug;
  fn exhaustive_hash_eq(&self, string_interner: &StringInterner) -> Box<dyn Iterator<Item = Self::Item>>;
  //fn create_hash_eq(bools: &[(u32, bool)], integers: &[(u32, i64)], strings: &[(u32, StrSym)]) -> Self::Item;
}

pub trait FactFn<T: ?Sized> {
  fn get_field(&self, idx: u32) -> &T;
  fn set_field(&mut self, idx: u32, to: &T);
}

pub trait FactDef {
  fn type_id(&self) -> TypeId;
  fn name(&self) -> &str;
  fn module(&self) -> &str;
  fn bool_fields(&self) -> &[(&str, u32)];
  fn i64_fields(&self) -> &[(&str, u32)];
  fn f64_fields(&self) -> &[(&str, u32)];
  fn str_fields(&self) -> &[(&str, u32)];
}

#[derive(Clone, Debug)]
pub struct StaticFactDef {
  type_id: TypeId,
  name: &'static str,
  module: &'static str,
  bool_fields: &'static [(&'static str, u32)],
  i64_fields: &'static [(&'static str, u32)],
  real_fields: &'static [(&'static str, u32)],
  str_fields: &'static [(&'static str, u32)]
}

impl StaticFactDef {
  pub fn new( type_id: TypeId,
               name: &'static str,
               module: &'static str,
               bool_fields: &'static [(&'static str, u32)],
               i64_fields: &'static [(&'static str, u32)],
               real_fields: &'static [(&'static str, u32)],
               str_fields: &'static [(&'static str, u32)]) -> StaticFactDef {
    StaticFactDef{
      type_id,
      name,
      module,
      bool_fields,
      i64_fields,
      real_fields,
      str_fields
    }
  }
}

impl FactDef for StaticFactDef {
  fn type_id(&self) -> TypeId {
    self.type_id
  }

  fn name(&self) -> &str {
    self.name
  }

  fn module(&self) -> &str {
    self.module
  }

  fn bool_fields(&self) -> &[(&str, u32)] {
    self.bool_fields
  }

  fn i64_fields(&self) -> &[(&str, u32)] {
    self.i64_fields
  }

  fn f64_fields(&self) -> &[(&str, u32)] {
    self.real_fields
  }

  fn str_fields(&self) -> &[(&str, u32)] {
    self.str_fields
  }
}

pub trait GetStaticFactDef {
  fn get_static_fact_def() -> StaticFactDef where Self: Sized;
}

pub trait PrimFactFn: FactFn<bool> + FactFn<i64> + FactFn<f64> + FactFn<str> {}

impl<T> PrimFactFn for T
  where T: FactFn<bool> + FactFn<i64> + FactFn<f64> + FactFn<str> {}

pub trait StaticFact: GetStaticFactDef + StaticFactHashEq + PrimFactFn + Any + Clone {}

impl<T> StaticFact for T
  where T: GetStaticFactDef + StaticFactHashEq + PrimFactFn + Any + Clone {}

pub trait DynamicFact: PrimFactFn + Any + DynClone {
  fn type_id(&self) -> TypeId;
}

impl<T> DynamicFact for T
  where T: PrimFactFn + Any + DynClone {
  fn type_id(&self) -> TypeId {
    std::any::TypeId::of::<T>()
  }
}

pub trait MetisFact: StaticFact + Into<Box<dyn DynamicFact>> {}

impl<T> From<T> for Box<dyn DynamicFact>
  where T: DynamicFact {
  fn from(t: T) -> Self {
    Box::new(t)
  }
}

impl<T> MetisFact for T
  where T: StaticFact + Into<Box<dyn DynamicFact>> {}

pub trait STest<T: ?Sized> {
  fn test(self, val: &T, to: &T) -> bool;
}

#[derive(Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub enum Truth {
  Not,
  Is,
}

impl Truth {
  pub fn is_not(self) -> bool {
    use self::Truth::*;
    match self {
      Not => true,
      Is => false,
    }
  }
}

/// Single value equivalence test
#[derive(Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub enum EqTest {
  /// val == to
  Eq,
  /// val != to
  Ne,
}

impl<T> STest<T> for EqTest
  where
    T: Eq + ?Sized,
{
  fn test(self, val: &T, to: &T) -> bool {
    use self::EqTest::*;
    match self {
      Eq => val == to,
      Ne => val != to,
    }
  }
}

/// Single value ordinal test
#[derive(Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub enum OrdTest {
  /// val < to
  Lt,
  /// val <= to
  Le,
  /// val > to
  Gt,
  ///
  /// val >= to
  Ge,
}

impl<T> STest<T> for OrdTest
  where
    T: Ord + ?Sized,
{
  fn test(self, val: &T, to: &T) -> bool {
    use self::OrdTest::*;
    match self {
      Lt => val < to,
      Le => val <= to,
      Gt => val > to,
      Ge => val >= to,
    }
  }
}

/// Single value approximate equivalence test for floats (default to 4 ULPs)
#[derive(Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub enum ApproxEqTest {
  /// val ~= to
  Eq,
  /// val !~= to
  Ne,
}

impl From<EqTest> for ApproxEqTest {
  fn from(eq: EqTest) -> ApproxEqTest {
    match eq {
      EqTest::Eq => ApproxEqTest::Eq,
      EqTest::Ne => ApproxEqTest::Ne,
    }
  }
}

impl<T> STest<T> for ApproxEqTest
  where T: Float + ApproxEq + ?Sized {
  fn test(self, val: &T, to: &T) -> bool {
    use self::ApproxEqTest::*;
    match self {
      Eq => val.approx_eq(*to, T::Margin::default()),
      Ne => val.approx_ne(*to, T::Margin::default())
    }
  }
}
/// &str tests
#[derive(Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub enum StrTest {
  /// val.contains(to)
  Contains,
  /// val.starts_with(to)
  StartsWith,
  /// val.ends_with(to)
  EndsWith
}

impl<T> STest<T> for StrTest
  where T: AsRef<str> + ?Sized {
  fn test(self, val: &T, to: &T) -> bool {
    use self::StrTest::*;
    match self {
      Contains => val.as_ref().contains(to.as_ref()),
      StartsWith => val.as_ref().starts_with(to.as_ref()),
      EndsWith => val.as_ref().ends_with(to.as_ref()),
    }
  }
}

#[derive(Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct HashEqItemIterator<T> {
  display_value: bool,
  display_none: bool,
  item: Option<T>
}

impl<T:Copy + Eq + Hash> HashEqItemIterator<T> {

  pub fn new(t: Option<T>) -> HashEqItemIterator<T> {
    HashEqItemIterator{display_value: t.is_some(), display_none: true, item: t}
  }

  pub fn some(t: T) -> HashEqItemIterator<T> {
    Self::new(Some(t))
  }

  pub fn none() -> HashEqItemIterator<T> {
    Self::new(None)
  }

}

impl<T:Copy + Eq + Hash> Iterator for HashEqItemIterator<T> {
  type Item = Option<T>;

  fn next(&mut self) -> Option<Self::Item> {
    if self.display_value {
      self.display_value = false;
      return Some(self.item);
    } else if self.display_none {
      self.display_none = false;
      return Some(None);
    }
    None
  }
}

impl<T: Copy + Eq + Hash> FusedIterator for HashEqItemIterator<T> {}

impl<T: Copy + Eq + Hash> ExactSizeIterator for HashEqItemIterator<T> {
    fn len(&self) -> usize {
      if self.display_value {
        return 2;
      } else if self.display_none {
        return 1;
      }
      0
    }
}

#[derive(Copy, Clone, Eq, Hash, Ord, PartialOrd, PartialEq, Debug)]
pub struct SerialGen<T: Unsigned, R> {
  r: PhantomData<R>,
  next: T
}

impl<T: Unsigned + One + Copy, R: From<T>> SerialGen<T, R> {
  pub fn new(next: T) -> Self {
    SerialGen{r: PhantomData, next}
  }

  pub fn idx(&self) -> R {
    let current = self.next - T::one();
    current.into()
  }
}

impl<T: Unsigned + One + Copy, R: From<T>> Iterator for SerialGen<T, R> {
  type Item = R;

  fn next(&mut self) -> Option<Self::Item> {
    let current = self.next;
    self.next = current + T::one();
    Some(current.into())
  }
}

impl<T: Unsigned + Zero + Copy, R: From<T>> Default for SerialGen<T, R> {
  fn default() -> Self {
    SerialGen::new(T::zero())
  }
}

macro_rules! serial_id_generator {
  ( $( $t:ident => $r:ident => $s:ident),+ ) => {
    $(
      #[derive(Copy, Clone, Eq, Hash, Ord, PartialOrd, PartialEq, Debug)]
      pub struct $r{idx: $t}

      impl From<$t> for $r {
        fn from(t: $t) -> Self {
           $r{idx: t}
        }
      }

      impl From<$r> for $t {
        fn from(r: $r) -> Self {
          r.idx
        }
      }

      pub type $s = SerialGen<$t, $r>;
    )*
  };
}

serial_id_generator!(
  u32 => FieldId => FieldIdGen,
  usize => StrSym => StrSymGen
);

impl Symbol for StrSym {
  fn from_usize(val: usize) -> Self {
    val.into()
  }

  fn to_usize(self) -> usize {
    self.idx.into()
  }
}

pub type StringInterner = string_interner::StringInterner<StrSym>;
