#ifndef DUAL_OUTPUT_H
#define DUAL_OUTPUT_H

#include <iostream>
#include <fstream>
#include <streambuf>

class dual_outputbuf : public std::streambuf {
public:
    dual_outputbuf(std::streambuf* buf1, std::streambuf* buf2)
        : buf1(buf1), buf2(buf2) {}

protected:
    virtual int overflow(int c) {
        if (c == EOF) {
            return !EOF;
        }
        else {
            int const r1 = buf1->sputc(c);
            int const r2 = buf2->sputc(c);
            return r1 == EOF || r2 == EOF ? EOF : c;
        }
    }

    virtual int sync() {
        int const r1 = buf1->pubsync();
        int const r2 = buf2->pubsync();
        return r1 == 0 && r2 == 0 ? 0 : -1;
    }

private:
    std::streambuf* buf1;
    std::streambuf* buf2;
};

class dual_output : public std::ostream {
public:
    dual_output(std::ostream& o1, std::ostream& o2)
        : std::ostream(&buf), buf(o1.rdbuf(), o2.rdbuf()) {}

private:
    dual_outputbuf buf;
};

#endif // DUAL_OUTPUT_H