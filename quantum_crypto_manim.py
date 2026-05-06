from manim import *


class DefaultTemplate(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set color and transparency

        square = Square()  # create a square
        square.flip(RIGHT)  # flip horizontally
        square.rotate(-3 * TAU / 8)  # rotate a certain amount

        self.play(Create(square))  # animate the creation of the square
        self.play(
            Transform(square, circle)
        )  # interpolate the square into the circle
        self.play(FadeOut(square))  # fade out animation


latex_preamble = r"""\usepackage{mathtools}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows}
\DeclarePairedDelimiter\bra{\langle}{\rvert}
\DeclarePairedDelimiter\ket{\lvert}{\rangle}
\DeclarePairedDelimiterX\braket[2]{\langle}{\rangle}{#1\,\delimsize\vert\,\mathopen{}#2}"""
myTemplate = TexTemplate()
myTemplate.add_to_preamble(latex_preamble)


class DefinitionOfMeasurementOperators(Scene):
    def construct(self):
        plus_basis = MathTex(
            r"""
            \mathcal{B}_+ &= \{\ket{0}, \ket{\pi}\} \\
            \braket{0}{\pi} &= 0 \\
            \braket{0}{0} &= 1 \\
            \braket{\pi}{\pi} &= 1
            """,
            tex_template=myTemplate,
        )
        m_plus_eigenvectors = MathTex(
            r"""
            M_+ \ket{0} &= -\ket{0} \\
            M_+ \ket{\pi} &= \ket{\pi} \\
            """,
            tex_template=myTemplate,
        )
        times_basis_definition = MathTex(
            r"""
            \ket{-\frac \pi 2} &= \frac 1 \sqrt{2} \ket{0} - \frac 1 \sqrt{2} \ket{pi} \\
            \ket{\frac \pi 2} &= \frac 1 \sqrt{2} \ket{0} + \frac 1 \sqrt{2} \ket{\pi}
            """
        )
        m_times_eigenvectors = MathTex(
            r"""
            M_\times \ket{-\frac \pi 2} &= \ket{-\frac \pi 2} \\
            M_\times \ket{\frac \pi 2} &= \ket{\pi 2} \\
            """,
            tex_template=myTemplate,
        )
        times_basis = MathTex(
            r"""
            \mathcal{B}_\times &= \{\ket{-\frac \pi 2}, \ket{\frac \pi 2}\} \\
            \braket{-\frac \pi 2}{\frac \pi 2} &= 0 \\
            \braket{-\frac \pi 2}{-\frac \pi 2} &= 1 \\
            \braket{\frac \pi 2}{\frac \pi 2} &= 1
            """,
            tex_template=myTemplate,
        )


class ConstructionOfMOperators(Scene):
    def construct(self):
        m_plus_operator = MathTex(
            r"M_+ = (\ket{\pi}\bra{\pi} - \ket{0}\bra{0})",
            tex_template=myTemplate,
        )
        plugging_into_m_plus = MathTex(
            r"""
            (\ket{\pi}\bra{\pi} - \ket{0}\bra{0}) \ket{0} &= (\ket{\pi}\bra{\pi}) \ket{0} - (\ket{0}\bra{0}) \ket{0} = (-1) \ket{0} \\
            (\ket{\pi}\bra{\pi} - \ket{0}\bra{0}) \ket{\pi} &= (\ket{\pi}\bra{\pi}) \ket{\pi} - (\ket{0}\bra{0}) \ket{\pi} = (+1) \ket{\pi}
            """,
            tex_template=myTemplate,
        )
        linearity_of_m_plus = MathTex(
            r"""
            (\ket{\pi}\bra{\pi} - \ket{0}\bra{0}) (\alpha\ket{a} + \beta\ket{b}) = \\
            \ket{\pi}\bra{\pi}\alpha\ket{a} - \ket{0}\bra{0}\beta\ket{b} + \ket{\pi}\bra{\pi}\alpha\ket{a} - \ket{0}\bra{0}\beta\ket{b} = \\
            (\ket{\pi}\bra{\pi} - \ket{0}\bra{0}) \alpha\ket{a} + (\ket{\pi}\bra{\pi} - \ket{0}\bra{0}) \beta\ket{b}
            """,
            tex_template=myTemplate,
        )
        m_times_operator = MathTex(
            r"M_\times = (\ket{\frac \pi 2}\bra{\frac \pi 2} - \ket{-\frac \pi 2}\bra{-\frac \pi 2})",
            tex_template=myTemplate,
        )


class EveDetectionProbability(Scene):
    def construct(self):
        flowchart = Tex(
            r"""
            \tikzstyle{line} = [draw,-latex]
            \begin{tikzpicture}[node distance=30mm]
                \node (title) {Eve's chances};
                \node (right_basis) [below right of=title,align=right,text width=3cm] {Choose the same basis as Alice and Bob};
                \node (wrong_basis) [below left of=title,align=left,text width=3cm] {Choose a different basis that Alice and Bob};
                \node (right_bit) [below right of=wrong_basis,align=right,text width=3cm] {Bob measures the same bit that Alice sent};
                \node (wrong_bit) [below left of=wrong_basis,align=left,text width=3cm] {Bob measures a different bit than Alice sent};
                \path [line] (title) -- node[anchor=south west] {$p=\frac 1 2$} (right_basis);
                \path [line,red] (title) -- node[anchor=south east] {$p=\frac 1 2$} (wrong_basis);
                \path [line] (wrong_basis) -- node[anchor=south west] {$p=\frac 1 2$} (right_bit);
                \path [line,red] (wrong_basis) -- node[anchor=south east] {$p=\frac 1 2$} (wrong_bit);
            \end{tikzpicture}
            """,
            tex_template=myTemplate,
        )

        self.add(flowchart)
        # chart_title = Text("Eve's decision tree", font_size=12)
        # chart_title_outline = SurroundingRectangle(chart_title)
        ## undetected, 50%
        # decisions = VGroup(*[

        #    ])
        # right_basis = Text(
        #    "Choose the same basis as Alice and Bob", font_size=12
        # )
        ## 50%
        # wrong_basis = Text(
        #    "Choose a different basis than Alice and Bob", font_size=12
        # )
        ## undetected, 50%
        # right_bit = Text(
        #    "Bob measures the same bit that Alice sent", font_size=12
        # )
        ## detected, 50%
        # wrong_bit = Text(
        #    "Bob measures a different bit than Alice sent", font_size=12
        # )

        # chart_title.to_edge(UP)
        # right_basis.align_to(chart_title, DR)
        # wrong_basis.align_to(chart_title, DL)
        # right_bit.align_to(wrong_basis, DR)
        # wrong_bit.align_to(wrong_basis, DL)

        # self.play(
        #    FadeIn(chart_title),
        #    FadeIn(chart_title_outline),
        # )
        # self.wait()
        # self.play(
        #    FadeIn(right_basis),
        #    FadeIn(wrong_basis),
        # )

