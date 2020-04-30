extern crate proc_macro;

use std::collections::BTreeMap;
use std::process::id;

use syn::{braced, Error, Field, Ident, parse_macro_input, Result, token, Token};
use syn::export::{TokenStream, Span};
use syn::parse::{Parse, ParseBuffer, ParseStream};
use syn::punctuated::Punctuated;
use syn::token::Token;

use quote::quote;

#[derive(Clone, Debug)]
struct StaticFactMacroInput {
  fact_name: Ident,
  bool_fields : BTreeMap<u32, Ident>,
  i64_fields : BTreeMap<u32, Ident>,
  f64_fields: BTreeMap<u32, Ident>,
  string_fields : BTreeMap<u32, Ident>
}

impl Parse for StaticFactMacroInput {
  fn parse<'a>(input: &'a ParseBuffer<'a>) -> Result<Self> {
    let content;
    let _struct_token: Token![struct] = input.parse()?;
    let ident: Ident = input.parse()?;
    let _brace_token: token::Brace = braced!(content in input);
    let fields: Punctuated<Field, Token![,]> = content.parse_terminated(Field::parse_named)?;

    let mut bool_fields : BTreeMap<u32, Ident> = BTreeMap::new();
    let mut i64_fields : BTreeMap<u32, Ident> = BTreeMap::new();
    let mut f64_fields: BTreeMap<u32, Ident> = BTreeMap::new();
    let mut string_fields : BTreeMap<u32, Ident> = BTreeMap::new();
    let mut idx = 0;
    for ref field in fields {
      use syn::Type::Path;
      match (&field.ident, &field.ty) {
        (Some(ref ident), Path(ref p))=> {
          if p.path.is_ident("bool") {
            bool_fields.insert(idx, ident.clone());
          } else if p.path.is_ident("i64") {
            i64_fields.insert(idx, ident.clone());
          } else if p.path.is_ident("f64") {
            f64_fields.insert(idx, ident.clone());
          } else if p.path.is_ident("String") {
            string_fields.insert(idx, ident.clone());
          }
          idx += 1;
        },
        _ => {}
      }
    }
    Ok(StaticFactMacroInput{fact_name: ident, bool_fields, i64_fields, f64_fields, string_fields})
  }
}

impl Into<proc_macro::TokenStream> for StaticFactMacroInput {
  fn into(self) -> TokenStream {
    let ref fact_name = self.fact_name;
    let fact_name_string = fact_name.to_string();


    let bool_idx_get_iter = self.bool_fields.iter().map(|f| f.0);
    let bool_idx_set_iter = self.bool_fields.iter().map(|f| f.0);
    let bool_field_get_iter = self.bool_fields.iter().map(|f| f.1);
    let bool_field_set_iter = self.bool_fields.iter().map(|f| f.1);

    let mut fact_fn_bool = quote! {

        impl metis_core::FactFn<bool> for #fact_name {
          fn get_field(&self, idx: u32) -> &bool {
            match idx {
              #(#bool_idx_get_iter => &self.#bool_field_get_iter),*,
              _ => panic!("Unexpected get access for bool {}", idx)
            }
          }

          fn set_field(&mut self, idx: u32, to: &bool) {
            match idx {
              #(#bool_idx_set_iter => self.#bool_field_set_iter = *to),*,
              _ => panic!("Unexpected set access for bool {}", idx)

            }
          }
        }
      };

    let i64_idx_get_iter = self.i64_fields.iter().map(|f| f.0);
    let i64_idx_set_iter = self.i64_fields.iter().map(|f| f.0);
    let i64_field_get_iter = self.i64_fields.iter().map(|f| f.1);
    let i64_field_set_iter = self.i64_fields.iter().map(|f| f.1);

    let fact_fn_i64 = quote! {
        impl metis_core::FactFn<i64> for #fact_name {
          fn get_field(&self, idx: u32) -> &i64 {
            match idx {
              #(#i64_idx_get_iter => &self.#i64_field_get_iter),*,
              _ => panic!("Unexpected get access for i64 {}", idx)
            }
          }

          fn set_field(&mut self, idx: u32, to: &i64) {
            match idx {
              #(#i64_idx_set_iter => self.#i64_field_set_iter = *to),*,
              _ => panic!("Unexpected set access for i64 {}", idx)

            }
          }
        }
      };

    let f64_idx_get_iter = self.f64_fields.iter().map(|f| f.0);
    let f64_idx_set_iter = self.f64_fields.iter().map(|f| f.0);
    let f64_field_get_iter = self.f64_fields.iter().map(|f| f.1);
    let f64_field_set_iter = self.f64_fields.iter().map(|f| f.1);

    let fact_fn_f64 = quote! {
        impl metis_core::FactFn<f64> for #fact_name {
          fn get_field(&self, idx: u32) -> &f64 {
            match idx {
              #(#f64_idx_get_iter => &self.#f64_field_get_iter),*,
              _ => panic!("Unexpected get access for f64 {}", idx)
            }
          }

          fn set_field(&mut self, idx: u32, to: &f64) {
            match idx {
              #(#f64_idx_set_iter => self.#f64_field_set_iter = *to),*,
              _ => panic!("Unexpected set access for f64 {}", idx)

            }
          }
        }
      };

    let str_idx_get_iter = self.string_fields.iter().map(|f| f.0);
    let str_idx_set_iter = self.string_fields.iter().map(|f| f.0);
    let str_field_get_iter = self.string_fields.iter().map(|f| f.1);
    let str_field_set_iter = self.string_fields.iter().map(|f| f.1);

    let fact_fn_str = quote! {
        impl metis_core::FactFn<str> for #fact_name {
          fn get_field(&self, idx: u32) -> &str {
            match idx {
              #(#str_idx_get_iter => &self.#str_field_get_iter),*,
              _ => panic!("Unexpected get access for str {}", idx)
            }
          }

          fn set_field(&mut self, idx: u32, to: &str) {
            match idx {
              #(#str_idx_set_iter => self.#str_field_set_iter = to.into()),*,
              _ => panic!("Unexpected set access for str {}", idx)

            }
          }
        }
      };


    let bool_idx_list_iter = self.bool_fields.iter().map(|f| f.0);
    let bool_field_list_iter = self.bool_fields.iter().map(|f| f.1.to_string());
    let i64_idx_list_iter = self.i64_fields.iter().map(|f| f.0);
    let i64_field_list_iter = self.i64_fields.iter().map(|f| f.1.to_string());
    let f64_idx_list_iter = self.f64_fields.iter().map(|f| f.0);
    let f64_field_list_iter = self.f64_fields.iter().map(|f| f.1.to_string());
    let str_idx_list_iter = self.string_fields.iter().map(|f| f.0);
    let str_field_list_iter = self.string_fields.iter().map(|f| f.1.to_string());

    let static_fact = quote! {
      impl metis_core::GetStaticFactDef for #fact_name {
        fn get_static_fact_def() -> metis_core::StaticFactDef {
          metis_core::StaticFactDef::new(
            ::std::any::TypeId::of::<Test>(),
            &#fact_name_string,
            &"",
            &[#((#bool_field_list_iter, #bool_idx_list_iter)),*],
            &[#((#i64_field_list_iter, #i64_idx_list_iter)),*],
            &[#((#f64_field_list_iter, #f64_idx_list_iter)),*],
            &[#((#str_field_list_iter, #str_idx_list_iter)),*]
          )
        }
      }
    };

    let bool_ident = quote!{Option<bool>};
    let bool_idx_hasheq_item_iter = self.bool_fields.iter().map(|_| &bool_ident);
    let i64_ident = quote!{Option<i64>};
    let i64_idx_hasheq_item_iter = self.i64_fields.iter().map(|_| &i64_ident);
    let str_ident = quote!{Option<metis_core::StrSym>};
    let str_idx_hasheq_item_iter = self.string_fields.iter().map(|_| &str_ident);

    let bool_field_hasheq_var_iter = self.bool_fields.iter()
      .map(|f| f.1)
      .map(|i| quote!{let #i = metis_core::HashEqItemIterator::some(self.#i);});
    let i64_field_hasheq_var_iter = self.i64_fields.iter()
      .map(|f| f.1)
      .map(|i| quote!{let #i = metis_core::HashEqItemIterator::some(self.#i);});
    let str_field_hasheq_var_iter = self.string_fields.iter()
      .map(|f| f.1)
      .map(|i| quote!{let #i = metis_core::HashEqItemIterator::new(string_interner.get(&self.#i));});

    let bool_field_hasheq_tuple_iter = self.bool_fields.iter().map(|f| f.1);
    let i64_field_hasheq_tuple_iter = self.i64_fields.iter().map(|f| f.1);
    let str_field_hasheq_tuple_iter = self.string_fields.iter().map(|f| f.1);

    let fact_hash_eq = quote! {
      impl metis_core::StaticFactHashEq for #fact_name {
        type Item = ( #(#i64_idx_hasheq_item_iter),* , #(#str_idx_hasheq_item_iter),* , #(#bool_idx_hasheq_item_iter),*);

        fn exhaustive_hash_eq(&self, string_interner: &metis_core::StringInterner) -> Box<dyn Iterator<Item = Self::Item>> {
          #(#bool_field_hasheq_var_iter)*
          #(#i64_field_hasheq_var_iter)*
          #(#str_field_hasheq_var_iter)*

          Box::new(itertools::iproduct!(#(#i64_field_hasheq_tuple_iter),* , #(#str_field_hasheq_tuple_iter),* , #(#bool_field_hasheq_tuple_iter),*))
        }
      }
    };

    let out_tokens = quote! {
      #fact_fn_bool
      #fact_fn_i64
      #fact_fn_f64
      #fact_fn_str

      #fact_hash_eq

      #static_fact
    };
    TokenStream::from(out_tokens)
  }
}


#[proc_macro_derive(MetisFact)]
pub fn metis_fact_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
  let input = parse_macro_input!(input as StaticFactMacroInput);
  input.into()
}
