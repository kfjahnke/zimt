namespace zimt_AVX2
{
  extern void foo() ;
} ;

namespace zimt_SSE
{
  extern void foo() ;
} ;

int main ( int argc , char * argv[] )
{
  zimt_SSE::foo() ;
  zimt_AVX2::foo() ;
}
