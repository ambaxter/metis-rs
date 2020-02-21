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
  integer_fields : BTreeMap<u32, Ident>,
  real_fields : BTreeMap<u32, Ident>,
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
    let mut integer_fields : BTreeMap<u32, Ident> = BTreeMap::new();
    let mut real_fields : BTreeMap<u32, Ident> = BTreeMap::new();
    let mut string_fields : BTreeMap<u32, Ident> = BTreeMap::new();
    let mut idx = 0;
    for ref field in fields {
      use syn::Type::Path;
      match (&field.ident, &field.ty) {
        (Some(ref ident), Path(ref p))=> {
          if p.path.is_ident("bool") {
            bool_fields.insert(idx, ident.clone());
          } else if p.path.is_ident("i64") {
            integer_fields.insert(idx, ident.clone());
          } else if p.path.is_ident("f64") {
            real_fields.insert(idx, ident.clone());
          } else if p.path.is_ident("String") {
            string_fields.insert(idx, ident.clone());
          }
          idx += 1;
        },
        _ => {}
      }
    }
    Ok(StaticFactMacroInput{fact_name: ident, bool_fields, integer_fields, real_fields, string_fields})
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

    let int_idx_get_iter = self.integer_fields.iter().map(|f| f.0);
    let int_idx_set_iter = self.integer_fields.iter().map(|f| f.0);
    let int_field_get_iter = self.integer_fields.iter().map(|f| f.1);
    let int_field_set_iter = self.integer_fields.iter().map(|f| f.1);

    let fact_fn_integer = quote! {
        impl metis_core::FactFn<i64> for #fact_name {
          fn get_field(&self, idx: u32) -> &i64 {
            match idx {
              #(#int_idx_get_iter => &self.#int_field_get_iter),*,
              _ => panic!("Unexpected get access for int {}", idx)
            }
          }

          fn set_field(&mut self, idx: u32, to: &i64) {
            match idx {
              #(#int_idx_set_iter => self.#int_field_set_iter = *to),*,
              _ => panic!("Unexpected set access for int {}", idx)

            }
          }
        }
      };

    let real_idx_get_iter = self.real_fields.iter().map(|f| f.0);
    let real_idx_set_iter = self.real_fields.iter().map(|f| f.0);
    let real_field_get_iter = self.real_fields.iter().map(|f| f.1);
    let real_field_set_iter = self.real_fields.iter().map(|f| f.1);

    let fact_fn_real = quote! {
        impl metis_core::FactFn<f64> for #fact_name {
          fn get_field(&self, idx: u32) -> &f64 {
            match idx {
              #(#real_idx_get_iter => &self.#real_field_get_iter),*,
              _ => panic!("Unexpected get access for real {}", idx)
            }
          }

          fn set_field(&mut self, idx: u32, to: &f64) {
            match idx {
              #(#real_idx_set_iter => self.#real_field_set_iter = *to),*,
              _ => panic!("Unexpected set access for real {}", idx)

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
    let int_idx_list_iter = self.integer_fields.iter().map(|f| f.0);
    let int_field_list_iter = self.integer_fields.iter().map(|f| f.1.to_string());
    let real_idx_list_iter = self.real_fields.iter().map(|f| f.0);
    let real_field_list_iter = self.real_fields.iter().map(|f| f.1.to_string());
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
            &[#((#int_field_list_iter, #int_idx_list_iter)),*],
            &[#((#real_field_list_iter, #real_idx_list_iter)),*],
            &[#((#str_field_list_iter, #str_idx_list_iter)),*]
          )
        }
      }
    };

    let bool_ident = quote!{Option<bool>};
    let bool_idx_hasheq_item_iter = self.bool_fields.iter().map(|_| &bool_ident);
    let integer_ident = quote!{Option<i64>};
    let integer_idx_hasheq_item_iter = self.integer_fields.iter().map(|_| &integer_ident);
    let str_ident = quote!{Option<metis_core::StrSym>};
    let str_idx_hasheq_item_iter = self.string_fields.iter().map(|_| &str_ident);

    let bool_field_hasheq_var_iter = self.bool_fields.iter()
      .map(|f| f.1)
      .map(|i| quote!{let #i = metis_core::HashEqItemIterator::some(self.#i);});
    let integer_field_hasheq_var_iter = self.integer_fields.iter()
      .map(|f| f.1)
      .map(|i| quote!{let #i = metis_core::HashEqItemIterator::some(self.#i);});
    let str_field_hasheq_var_iter = self.string_fields.iter()
      .map(|f| f.1)
      .map(|i| quote!{let #i = metis_core::HashEqItemIterator::new(string_interner.get(&self.#i));});

    let bool_field_hasheq_tuple_iter = self.bool_fields.iter().map(|f| f.1);
    let integer_field_hasheq_tuple_iter = self.integer_fields.iter().map(|f| f.1);
    let str_field_hasheq_tuple_iter = self.string_fields.iter().map(|f| f.1);

    let fact_hash_eq = quote! {
      impl metis_core::StaticFactHashEq for #fact_name {
        type Item = ( #(#integer_idx_hasheq_item_iter),* , #(#str_idx_hasheq_item_iter),* , #(#bool_idx_hasheq_item_iter),*);

        fn exhaustive_hash_eq(&self, string_interner: &metis_core::StringInterner) -> Box<dyn Iterator<Item = Self::Item>> {
          #(#bool_field_hasheq_var_iter)*
          #(#integer_field_hasheq_var_iter)*
          #(#str_field_hasheq_var_iter)*

          Box::new(itertools::iproduct!(#(#integer_field_hasheq_tuple_iter),* , #(#str_field_hasheq_tuple_iter),* , #(#bool_field_hasheq_tuple_iter),*))
        }
      }
    };

    let out_tokens = quote! {
      #fact_fn_bool
      #fact_fn_integer
      #fact_fn_real
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
