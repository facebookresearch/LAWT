# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from abc import ABC, abstractmethod
import numpy as np
import math


class Encoder(ABC):
    """
    Base class for encoders, encodes and decodes matrices
    abstract methods for encoding/decoding numbers
    """
    def __init__(self, max_dimension, precision):
        self.max_encoder_dimension = max_dimension
        self.float_precision = precision
        self.symbols = ["V" + str(i) for i in range(1, self.max_encoder_dimension + 1)]
        self.limit = -1

    @abstractmethod
    def write_float(self, val):
        pass

    @abstractmethod
    def parse_float(self, lst):
        pass

    def encode(self, matrix):
        lst = []
        l, c = np.shape(matrix)
        lst.append("V" + str(l))
        lst.append("V" + str(c))
        for line in matrix:
            for val in line:
                lst.extend(self.write_float(val))
        return lst

    def decode(self, lst):
        if len(lst) < 2 or lst[0][0] != "V" or lst[1][0] != "V":
            return None
        nr_lines = int(lst[0][1:])
        nr_cols = int(lst[1][1:])
        h = lst[2:]
        m = np.zeros((nr_lines, nr_cols), dtype=float)
        for i in range(nr_lines):
            for j in range(nr_cols):
                val, pos = self.parse_float(h)
                if np.isnan(val):
                    return None
                h = h[pos:]
                m[i, j] = val
        return m


class FPSymbol(Encoder):
    def __init__(self, params, precision, max_exponent):
        super().__init__(params.max_encoder_dimension, precision)
        self.max_exponent = max_exponent
        assert (self.float_precision + self.max_exponent) % 2 == 0
        self.symbols.extend(["NaN", "-NaN"])
        dig = 10 ** self.float_precision
        self.logrange = (self.float_precision + self.max_exponent) // 2
        self.base = 10 ** (self.logrange - self.float_precision)
        self.limit = 10 ** self.logrange
        self.output_length = 1
        # less than 1
        self.symbols.extend(["N" + str(i) + "e0" for i in range(-dig + 1, dig)])
        for i in range(self.max_exponent):
            for j in range(1, 10):
                for k in range(dig):
                    self.symbols.append("N" + str(j * dig + k) + "e" + str(i))
                    self.symbols.append("N-" + str(j * dig + k) + "e" + str(i))

    def write_float(self, value):
        if abs(value) > self.limit:
            return ["NaN"] if value > 0 else ["-NaN"]
        sign = -1 if value < 0 else 1
        v = abs(value) * self.base
        if v == 0:
            return ["N0e0"]
        e = int(math.log10(v))
        if e < 0:
            e = 0
        m = int(v * (10 ** (self.float_precision - e)) + 0.5)
        if m == 0:
            sign = 1
        if m == 1000:
            m = 100
            e += 1
        if e >= self.max_exponent:
            return ["NaN"] if value > 0 else ["-NaN"]
        pref = "N" if sign == 1 else "N-"
        return [pref + str(m) + "e" + str(e)]

    def parse_float(self, lst):
        if len(lst) == 0:
            return np.nan, 0
        if lst[0] == "NaN":
            return self.limit, 1
        if lst[0] == "-NaN":
            return -self.limit, 1
        if lst[0][0] != "N":
            return np.nan, 1
        m, e = lst[0][1:].split("e")
        v = (int(m) * (10 ** int(e))) / self.limit
        return v, 1


class FloatSymbol(Encoder):
    def __init__(self, params, prec):
        super().__init__(params.max_encoder_dimension, prec)
        max_token = 10 ** (prec + 1)
        self.symbols.extend(['N' + str(i) for i in range(-max_token, max_token + 1)])
        self.symbols.extend(['E' + str(i) for i in range(-100, 101)])
        self.limit = 10.0 ** 101
        self.output_length = 2

    def write_float(self, value):
        """
        Write a float number
        """
        precision = self.float_precision
        assert value not in [-np.inf, np.inf]
        m, e = (f"%.{precision}e" % value).split("e")
        i, f = m.split(".")
        i = i + f
        ipart = int(i)
        expon = int(e) - precision
        if expon < -100:
            ipart = 0
        if ipart == 0:
            expon = 0
        return ['N' + str(ipart), "E" + str(expon)]

    def parse_float(self, lst):
        """
        Parse a list that starts with a float.
        Return the float value, and the position it ends in the list.
        """
        if len(lst) < 2 or lst[0][0] != 'N' or lst[1][0] != 'E':
            return np.nan, 0
        try:
            mant = int(lst[0][1:])
            exp = int(lst[1][1:])
            value = mant * (10 ** exp)
        except Exception:
            return np.nan, 2
        return value, 2


class Positional(Encoder):
    def __init__(self, params, prec, base_int):
        super().__init__(params.max_encoder_dimension, prec)
        self.base = base_int

        self.symbols.extend([str(i) for i in range(self.base)])
        self.symbols.extend(['E' + str(i) for i in range(-100, 101)])
        self.limit = 10.0 ** 101
        # WARNING adjust this at some point
        self.output_length = 5 if base_int == 10 else 3

    def gobble_int(self, lst):
        res = 0
        i = 0
        for x in lst:
            if not (x.isdigit()):
                break
            res = res * self.base + int(x)
            i += 1
        return res, i

    def write_posint(self, value):
        if value == 0:
            return ["0"]
        seq = []
        v = value
        while v > 0:
            seq.append(str(v % self.base))
            v = v // self.base
        return seq[::-1]

    def write_float(self, value):
        """
        Write a float number
        """
        precision = self.float_precision
        assert value not in [-np.inf, np.inf]
        m, e = (f"%.{precision}e" % abs(value)).split("e")
        i, f = m.split(".")
        i = i + f
        ipart = int(i)
        fpart = 0
        expon = int(e) - precision
        if expon < -100:
            ipart = 0
            fpart = 0
        if ipart == 0 and fpart == 0:
            value = 0.0
            expon = 0
        res = ["+"] if value >= 0.0 else ["-"]
        res = res + self.write_posint(ipart)
        if fpart != 0:
            res.append('.')
            res = res + self.write_posint(fpart)
        return res + ["E" + str(expon)]
            
    def parse_float(self, lst):
        """
        Parse a list that starts with a float.
        Return the float value, and the position it ends in the list.
        """
        if len(lst) < 2 or lst[0] not in ["+", "-"]:
            return np.nan, 0
        sign = -1 if lst[0] == "-" else 1
        pos = 1
        mant, i = self.gobble_int(lst[pos:])
        if i == 0:
            return np.nan, pos
        pos += i
        if len(lst) > pos and lst[pos] == ".":
            pos += 1
            base_mul = 1.0
            mul, i = self.gobble_int(lst[pos:])
            mul *= base_mul
            mul /= 10 ** self.float_precision
            pos += i
            mant += mul
        mant *= sign
        if len(lst) > pos and lst[pos] == "10^":
            pos += 1
            if lst[pos] not in ["+", "-"]:
                return np.nan, pos
            signexp = -1 if lst[pos] == "-" else 1
            pos += 1
            exp, i = self.gobble_int(lst[pos:])
            exp *= signexp
            if i == 0:
                return np.nan, pos
            pos += i
        elif len(lst) > pos and lst[pos][0] == "E":
            exp = int(lst[pos][1:])
            pos += 1
        else:
            exp = 0
        try:
            value = mant * (10 ** exp)
        except Exception:
            return np.nan, pos
        return value, pos
