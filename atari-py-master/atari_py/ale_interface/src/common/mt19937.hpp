/* Copyright (C) 2019 by Nikita Kniazev
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
 * INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE.
 */

#ifndef NK_MT19937
#define NK_MT19937

#include <stddef.h>

#if defined(_MSC_VER) && (_MSC_VER < 1600)
typedef unsigned __int32 uint32_t;
#else
# include <stdint.h>
# include <inttypes.h>
#endif

class mt19937
{
public:
  typedef uint32_t result_type;

  static const size_t word_size = 32;
  static const size_t state_size = 624;
  static const size_t shift_size = 397;
  static const size_t mask_bits = 31;
  static const uint32_t xor_mask = 0x9908b0df;
  static const size_t tempering_u = 11;
  static const uint32_t tempering_d = 0xffffffff;
  static const size_t tempering_s = 7;
  static const uint32_t tempering_b = 0x9d2c5680;
  static const size_t tempering_t = 15;
  static const uint32_t tempering_c = 0xefc60000;
  static const size_t tempering_l = 18;
  static const uint32_t initialization_multiplier = 1812433253;
  static const uint32_t default_seed = 5489u;


  explicit mt19937(uint32_t value = default_seed) { seed(value); }

  void seed(uint32_t value = default_seed)
  {
    state[0] = value;
    for (uint32_t i = 1; i < state_size; ++i)
      state[i] = i + initialization_multiplier * (state[i - 1] ^ (state[i - 1] >> 30));
    index = 0;
  }

  result_type operator()()
  {
    uint32_t next = (index + 1) % state_size;

    uint32_t y = (0x80000000 & state[index]) | (0x7fffffff & state[next]);
    uint32_t r = state[index] = state[(index + shift_size) % state_size] ^ (y >> 1) ^ ((-(y & 1)) & xor_mask);
    r ^= r >> tempering_u & tempering_d;
    r ^= r << tempering_s & tempering_b;
    r ^= r << tempering_t & tempering_c;
    r ^= r >> tempering_l;

    index = next;

    return r;
  }

  result_type (min)() const { return 0; }
  result_type (max)() const { return 0xffffffffUL; }

  template <typename OStream>
  friend OStream& operator<<(OStream& os, mt19937 const& g)
  {
    typename OStream::fmtflags saved_flags = os.flags(os.dec);

    for (uint32_t i = g.index; i < state_size; ++i)
      os << g.state[i] << ' ';
    for (uint32_t i = 0; i < g.index; ++i)
      os << g.state[i] << ' ';

    os.flags(saved_flags);

    return os;
  }

  template <typename IStream>
  friend IStream& operator>>(IStream& is, mt19937& g)
  {
    typename IStream::fmtflags saved_flags = is.flags(is.dec | is.skipws);

    uint32_t tmp[state_size];
    for (uint32_t i = 0; i < state_size; ++i)
      is >> tmp[i];

    is.flags(saved_flags);

    if (is.good()) {
      for (uint32_t i = 0; i < state_size; ++i)
        g.state[i] = tmp[i];
      g.index = 0;
    }

    return is;
  }

private:
  uint32_t state[state_size];
  uint32_t index;
};

#endif // NK_MT19937
