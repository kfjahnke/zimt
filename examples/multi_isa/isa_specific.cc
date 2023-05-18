#include <zimt/zimt.h>

namespace zimt
{
  void foo()
  {
    typedef zimt::simdized_type < float , 16 > f16_t ;
    std::cout << "base vector size: "
              << sizeof ( f16_t::vec_t ) << std::endl ;
    std::cout << f16_t::iota() << std::endl ;
  }
} ;
