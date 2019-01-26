import re
class Rational:

    def __init__(self,numerator,denominator):
        self.denominator = denominator
        self.numerator = numerator
        if numerator==0:
            self.denominator=1

    def __add__(self, other):
        #assert isinstance(other,int)
        new_denominator = self.denominator
        new_numerator = other*self.denominator+self.numerator
        return Rational(new_numerator,new_denominator)

    def __truediv__(self, other):
        #assert isinstance(other,int)
        return Rational(self.numerator,self.denominator*other)

    def reduce(self):
        def gcd(a,b):
            if a<b:
                a,b = b,a
            mod = a%b
            while mod!=0:
                a,b = b,mod
                mod = a%b
            return b
        if self.numerator==0:
            self.denominator=1
        else:
            gcd_ = gcd(self.numerator,self.denominator)
            self.numerator//=gcd_
            self.denominator//=gcd_

    def __eq__(self, other):
        return self.denominator==other.denominator and self.numerator==other.numerator

class Solution:
    def isRationalEqual(self, S, T):
        """
        :type S: str
        :type T: str
        :rtype: bool
        """
        S_splits = S.split(".")
        T_splits = T.split(".")
        s_integer = self.integer_part(S_splits[0])
        t_integer = self.integer_part(T_splits[0])
        s_fraction = S_splits[-1] if len(S_splits) > 1 else ""
        t_fraction = T_splits[-1] if len(T_splits) > 1 else ""
        s_fraction = self.fraction_part(s_fraction)
        t_fraction = self.fraction_part(t_fraction)
        s_value = s_fraction+s_integer
        t_value = t_fraction+t_integer
        s_value.reduce()
        t_value.reduce()
        return s_value==t_value

    def integer_part(self, s):
        if s == "":
            return 0
        return int(s)

    def fraction_part(self, s):
        if s == "":
            return Rational(0,0)
        repet = re.search("\(\d+\)", s)
        if repet is None:
            return Rational(int(s),pow(10, len(s)))
        non_repet_end = s.index("(")
        non_repet = int(s[:non_repet_end]) if non_repet_end > 0 else 0
        multipier = pow(10, non_repet_end)
        repet = repet.group()[1:-1]
        repet = Rational(int(repet),(pow(10, len(repet)) - 1))
        return (repet+non_repet) / multipier

if __name__ == "__main__":
    a = "0.1666(6)"
    b = "0.166(66)"
    print(Solution().isRationalEqual(a,b))