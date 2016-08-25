
import sys
import argparse

from collections import namedtuple


class Column(namedtuple("Column", ["name", "width", "fmt", "right", "colspan"])):
    """Columns for a Markdown appropriate text table."""

    @property
    def header(self):
        align = "" if self.right else "-"
        header_fmt = " %" + align + str(self.width-2) + "s "
        return header_fmt % self.name

    @property
    def sub_header(self):
        val = "-" * (self.width-2)
        if self.right:
            val = " " + val + ":"
        else:
            val = ":" + val + " "
        return val

    def format(self, data):
        return self.fmt % data

Column.__new__.__defaults__ = (None, None, None, False, 1)

class Formatter(object):
    def __init__(self, columns):
        self.columns = columns

    def print_header(self):
        pass

    def print_footer(self):
        pass

class Text(Formatter):
    def print_header(self):
        print "|".join(c.header for c in self.columns)
        print "|".join(c.sub_header for c in self.columns)

    def print_row(self, data):
        print "|".join(c.fmt % data[i] for i, c in enumerate(self.columns))

class HTML(Formatter):
    def print_header(self):
        print "<table>"
        print "<tr>" + "".join('<th colspan="%d">%s</th>' % (c.colspan, c.header)
                               for c in self.columns) + "</tr>"

    def print_row(self, data):
        print "".join("<td>" + c.fmt % data[i] + "</td>"
                      for i, c in enumerate(self.columns))

    def print_footer(self):
        print "</table>"

def format_sci_latex(s):
    if "e+" in s or "e-" in s:
        k, exp = s.split("e")
        if exp[1].strip() == '0':
            exp = exp[0] + exp[2]
        if exp[0] == '+':
            exp = exp[1:]
        return r"$%s \times 10^{%s}$" % (k, exp)
    else:
        return s

class Latex(Formatter):
    def print_header(self):
        print r"\begin{tabular}"
        print "&".join("\multicolumn{%d}{c}{%s}" % (c.colspan, c.header)
                       if c.colspan != 1
                       else c.header
                       for c in self.columns) + r" \\"


    def print_row(self, data):
        print ("&".join(
            [r"\texttt{%s}" % data[0].replace("_", "\_")] +
            [format_sci_latex(c.fmt % data[i+1])
             for i, c in enumerate(self.columns[1:])])
        + r" \\")

    def print_footer(self):
        print r"\end{tabular}"


FORMATTERS = {
    "text": Text,
    "html": HTML,
    "latex": Latex,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", default="text")
    args = parser.parse_args()

    results = {}

    for line in sys.stdin:
        benchmark, problem, time, value = line.split()
        results[(problem, benchmark)] = (float(time), float(value))

    problems = set(k[0] for k in results)
    benchmarks = set(k[1] for k in results)

    columns = [Column("Problem",   18, "%-18s")]
    for benchmark in benchmarks:
        columns += [
            Column("Time",      8,  "%7.2fs", right=True),
            Column("Objective", 11, "%11.2e", right=True),
        ]

    formatter = FORMATTERS[args.format](columns)
    formatter.print_header()

    for problem in sorted(problems):
        row = [problem]
        for benchmark in ["epsilon", "scs", "ecos"]:
            if benchmark not in benchmarks:
                continue
            row += results.get(
                (problem, benchmark), (float("nan"), (float("nan"))))

        formatter.print_row(row)
    formatter.print_footer()
