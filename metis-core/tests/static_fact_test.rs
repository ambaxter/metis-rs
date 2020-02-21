#[macro_use] extern crate itertools;
use itertools::Itertools;
use metis_core_derive::MetisFact;
use std::collections::HashSet;
use metis_core::{EqTest, FactFn, STest, HashEqItemIterator, FieldIdGen, StringInterner, StaticFactHashEq, MetisFact, FactDef, DynamicFact};
use std::ops::Deref;
use ordered_float::NotNan;
use std::any::Any;
use std::hash::Hash;

#[derive(Clone, Debug, MetisFact)]
struct Test {
  a: bool,
  e: bool,
  b: i64,
  c: f64,
  d: String
}

#[derive(Clone, Debug)]
struct TestBuilder {
  set: HashSet<u32>
}

impl Default for TestBuilder {
  fn default() -> Self {
    TestBuilder {set: HashSet::new()}
  }
}

impl TestBuilder {

  fn add_type<T: metis_core::MetisFact>(&mut self) -> &mut TestBuilder {
    let fact_def = T::get_static_fact_def();
    fact_def.str_fields()
      .iter()
      .map(|f| f.1)
      .for_each(|t| {
        self.set.insert(t);
      });
    println!("{:?}", fact_def.str_fields());
    self
  }
}


struct ConstField<T>{
  c: T,
  f_idx: u32
}

struct FieldConst<T> {
  f_idx: u32,
  c: T
}

enum AlphaNode{
  BCF(ConstField<bool>),
  BFC(FieldConst<bool>),
  ICF(ConstField<i64>),
  IFC(FieldConst<i64>),
  FCF(ConstField<f64>),
  FFC(FieldConst<f64>),
  SCF(ConstField<String>),
  SFC(FieldConst<String>),

}

#[test]
fn it_works() {
  println!("Size:{:?}", std::mem::size_of::<AlphaNode>());
  let mut test: Test = Test { a: false, e: true, b: 1, c: 1.0f64, d: "Hold me".to_string() };

  assert_eq!(false, EqTest::Eq.test(&true, FactFn::get_field(&test, 0)));
  FactFn::set_field(&mut test, 0, &true);
  assert_eq!(true, EqTest::Eq.test(&true, FactFn::get_field(&test, 0)));

  let mut test_box: Box<dyn DynamicFact> = Test { a: true, e: false, b: 0, c: 0.0f64, d: "nono".into() }.into();

  assert_eq!(&0, FactFn::<i64>::get_field(test_box.deref(), 2));

  let mut m : TestBuilder = Default::default();
  m
    .add_type::<Test>();

  println!("{:?}", m);
}

trait HashEq: Hash + Eq {}

pub fn runrunrun<T: MetisFact>(t: T) {
  let mut s = StringInterner::new();
  s.get_or_intern("nono");
  for i in t.exhaustive_hash_eq(&s) {
    println!("{:?}", i);
  }
}

#[test]
pub fn it_works_2() {
  let mut test_box = Test { a: true, e: false, b: 0, c: 0.0f64, d: "nono".into() };

  runrunrun(test_box);
}
