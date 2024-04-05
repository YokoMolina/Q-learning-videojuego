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

#ifndef NK_SCOPED_PTR
#define NK_SCOPED_PTR

namespace utils {

struct bool_conversion { int padding; int truth; };
typedef int bool_conversion::* explicit_bool;

}

template<class T>
class scoped_ptr {
public:
  typedef T*  pointer;
  typedef T   element_type;

  scoped_ptr() throw() : p_() {}
  explicit scoped_ptr(pointer p) throw() : p_(p) {}
  ~scoped_ptr() { destroy(); }

  element_type& operator*() const { return *get(); }
  pointer operator->() const throw() { return get(); }
  pointer get() const throw() { return p_; }
  operator bool() const throw() { return get() != pointer(); }
  operator utils::explicit_bool() const
  {
    return get() != pointer() ? &utils::bool_conversion::truth
                              : utils::explicit_bool(0);
  }

  pointer release() throw() { pointer p = p_; p_ = pointer(); return p; }
  void reset(pointer p = pointer()) throw() { destroy(); p_ = p; }
  void swap(scoped_ptr& u) throw()
  {
    pointer p = release();
    p_ = u.release();
    u.p_ = p;
  }

private:
  scoped_ptr(const scoped_ptr&);
  scoped_ptr& operator=(const scoped_ptr&);

  void destroy() { delete get(); }

  pointer p_;
};

#endif // ALE_SCOPED_PTR
