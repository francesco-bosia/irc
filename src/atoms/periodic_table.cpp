#include "periodic_table.h"

#include <iomanip>

namespace periodic_table{

bool valid_atomic_number(size_t an){
  return an > 0 && an < pt_size;
}

size_t atomic_number(const std::string &symbol) {
  size_t an{0};
  
  for(size_t i = 0; i < pt_size; i++){
    if(symbol == pt_symbols[i]){
      an = i;
      break;
    }
  }
  
  return an;
}

}