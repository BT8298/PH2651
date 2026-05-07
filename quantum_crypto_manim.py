import random
import itertools
from manim import *

latex_preamble = r"""\usepackage{mathtools}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows}
\DeclarePairedDelimiter\bra{\langle}{\rvert}
\DeclarePairedDelimiter\ket{\lvert}{\rangle}
\DeclarePairedDelimiterX\braket[2]{\langle}{\rangle}{#1\,\delimsize\vert\,\mathopen{}#2}"""
myTemplate = TexTemplate()
myTemplate.add_to_preamble(latex_preamble)


class DefinitionOfMOperators(Scene):
    def construct(self):
        hilbert_space = MathTex(r"\mathcal{H}")
        hilbert_space_dim = MathTex(r"\dim \mathcal{H} = 2")
        plus_basis = MathTex(
            r"""
            \mathcal{B}_+ &= \{\ket{0}, \ket{\pi}\} \\
            """,
            # \braket{0}{\pi} &= 0 \\
            # \braket{0}{0} &= 1 \\
            # \braket{\pi}{\pi} &= 1
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
            \left\lvert {-\frac \pi 2} \right\rangle &= \frac{1}{\sqrt{2}} \ket{0} - \frac{1}{\sqrt{2}} \ket{\pi} \\
            \left\lvert {\frac \pi 2} \right\rangle &= \frac{1}{\sqrt{2}} \ket{0} + \frac{1}{\sqrt{2}} \ket{\pi}
            """,
            tex_template=myTemplate,
        )
        m_times_eigenvectors = MathTex(
            r"""
            M_\times \left\lvert {-\frac \pi 2} \right\rangle &= -\left\lvert {-\frac \pi 2} \right\rangle \\
            M_\times \left\lvert {\frac \pi 2} \right\rangle &= \left\lvert {\frac \pi 2} \right\rangle \\
            """,
            tex_template=myTemplate,
        )
        times_basis = MathTex(
            r"""
            \mathcal{B}_\times &= \left\{\left\lvert {-\frac \pi 2} \right\rangle, \left\lvert {\frac \pi 2} \right\rangle \right\} \\
            """,
            # \braket{-\frac \pi 2}{\frac \pi 2} &= 0 \\
            # \braket{-\frac \pi 2}{-\frac \pi 2} &= 1 \\
            # \braket{\frac \pi 2}{\frac \pi 2} &= 1
            tex_template=myTemplate,
        )

        self.play(Write(hilbert_space))
        self.play(TransformMatchingTex(hilbert_space, hilbert_space_dim))
        self.play(hilbert_space_dim.animate.to_corner(UL))
        self.play(FadeIn(plus_basis))
        self.play(
            plus_basis.animate.next_to(hilbert_space_dim, DOWN).to_edge(LEFT)
        )
        self.play(FadeIn(m_plus_eigenvectors))
        self.play(m_plus_eigenvectors.animate.to_corner(UR))
        self.play(FadeIn(times_basis_definition))
        self.play(times_basis_definition.animate.to_corner(DL))
        self.play(FadeIn(times_basis))
        self.play(times_basis.animate.next_to(plus_basis, DOWN).to_edge(LEFT))
        self.play(FadeIn(m_times_eigenvectors))
        self.play(
            m_times_eigenvectors.animate.next_to(
                m_plus_eigenvectors, DOWN
            ).to_edge(RIGHT)
        )
        self.wait(3)


class ConstructionOfMOperators(Scene):
    def construct(self):
        m_plus_operator = MathTex(
            r"M_+ = \ket{\pi}\bra{\pi} - \ket{0}\bra{0}",
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
            &(\ket{\pi}\bra{\pi} - \ket{0}\bra{0}) (\alpha\ket{a} + \beta\ket{b}) = \\
            &\ket{\pi}\bra{\pi}\alpha\ket{a} + \ket{\pi}\bra{\pi}\beta\ket{b} - \ket{0}\bra{0}\alpha\ket{a} - \ket{0}\bra{0}\beta\ket{b} = \\
            &(\ket{\pi}\bra{\pi} - \ket{0}\bra{0}) (\alpha\ket{a}) + (\ket{\pi}\bra{\pi} - \ket{0}\bra{0}) (\beta\ket{b})
            """,
            tex_template=myTemplate,
        )
        linearity_steps = [
            MathTex(
                r"(\ket{\pi}\bra{\pi} - \ket{0}\bra{0}) (\alpha\ket{a} + \beta\ket{b})",
                tex_template=myTemplate,
            ),
            MathTex(
                r"(\ket{\pi}\bra{\pi})\alpha\ket{a} + (\ket{\pi}\bra{\pi})\beta\ket{b} - (\ket{0}\bra{0})\alpha\ket{a} - (\ket{0}\bra{0})\beta\ket{b}",
                tex_template=myTemplate,
            ),
            MathTex(
                r"(\ket{\pi}\bra{\pi} - \ket{0}\bra{0}) \alpha\ket{a} + (\ket{\pi}\bra{\pi} - \ket{0}\bra{0}) \beta\ket{b}",
                tex_template=myTemplate,
            ),
        ]
        m_times_operator = MathTex(
            r"M_\times = \left\lvert {\frac \pi 2} \right\rangle \left\langle {\frac \pi 2} \right\lvert - \left\lvert {-\frac \pi 2} \right\rangle \left\langle {-\frac \pi 2} \right\lvert",
            tex_template=myTemplate,
        )

        self.play(Write(m_plus_operator))
        self.wait(3)
        self.play(m_plus_operator.animate.to_corner(UL))
        self.play(FadeIn(plugging_into_m_plus))
        self.wait(3)
        self.play(FadeOut(plugging_into_m_plus))
        for step, nextstep in itertools.pairwise(linearity_steps):
            nextstep.next_to(step, DOWN)
            self.play(FadeIn(step))
        self.play(FadeIn(linearity_steps[-1]))
        # self.play(FadeIn(linearity_of_m_plus))
        self.wait(3)
        self.play(*[FadeOut(step) for step in linearity_steps])
        # self.play(FadeOut(linearity_of_m_plus))
        m_times_operator.next_to(m_plus_operator, DOWN)
        m_times_operator.to_edge(LEFT)
        self.play(Write(m_times_operator))
        self.wait(3)


# class Observation(Scene):
#    def construct(self):
#        al = Text('Alice uses + basis to send')
#        pi = MathTex(r'\ket{\pi}')
#        ev = Text('Eve measures in x basis:')
#        proj = MathTex(r'(\ket{\frac \pi 2}\bra{\frac \pi 2} - \ket{-\frac \pi 2}\bra{-\frac \pi 2}) \ket{\pi}')
#        transformed
#        diag = MathTex(r'\ket{\frac pi 2}')
#        diag2 = MathTex(r'\frac{1}{\sqrt{2}} \ket{0} + \frac{1}{\sqrt{2}} \ket{\pi}')
#
#        pi.next_to(al, RIGHT)

# class Observation(Scene):
#    pass


class BulkObservation(Scene):
    def construct(self):
        z = MathTex(r"(-1)\ket{0}", tex_template=myTemplate)
        p = MathTex(r"(+1)\ket{\pi}", tex_template=myTemplate)
        states = VGroup(
            *[
                MathTex(
                    r"\frac{1}{\sqrt{2}} \ket{0} + \frac{1}{\sqrt{2}} \ket{\pi}",
                    tex_template=myTemplate,
                ).scale(0.5)
                for i in range(0, 25)
            ]
        )
        states.arrange_in_grid()
        states.height = config.frame_height
        states.width = config.frame_width
        collapsed_states = [
            random.choice([z, p]).copy().move_to(state.get_center())
            for state in states.submobjects
        ]

        self.play(FadeIn(states))
        self.wait(3)
        self.play(
            *[
                Transform(state, collapsed_state)
                for state, collapsed_state in zip(
                    states.submobjects, collapsed_states
                )
            ]
        )
        self.wait(3)


class EveDetectionProbability(Scene):
    def construct(self):
        # flowchart = Tex(
        #    r"""
        #    \begin{tikzpicture}
        #        \tikzset{every picture/.style={line width=6pt}}
        #        \tikzstyle{line}=[draw]
        #        \tikzstyle{every node}=[draw]
        #        \node (title) {Eve's chances};
        #        \node (right_basis) [below right=of title,text width=3cm] {Choose the same basis as Alice and Bob};
        #        \node (wrong_basis) [below left=of title,text width=3cm] {Choose a different basis that Alice and Bob};
        #        \node (right_bit) [below right=of wrong_basis,text width=3cm] {Bob measures the same bit that Alice sent};
        #        \node (wrong_bit) [below left=of wrong_basis,text width=3cm] {Bob measures a different bit than Alice sent};
        #        \path [line] (title) -- node[anchor=south west,draw=none] {$p=\frac 1 2$} (right_basis);
        #        \path [line,red] (title) -- node[anchor=south east,draw=none] {$p=\frac 1 2$} (wrong_basis);
        #        \path [line] (wrong_basis) -- node[anchor=south west,draw=none] {$p=\frac 1 2$} (right_bit);
        #        \path [line,red] (wrong_basis) -- node[anchor=south east,draw=none] {$p=\frac 1 2$} (wrong_bit);
        #    \end{tikzpicture}
        #    """,
        #    tex_template=myTemplate,
        #    stroke_width=1
        # )

        wrong_basis = Text("Choose different basis than Alice and Bob")
        wrong_bit = Text("Happen to measure the wrong bit")
        detected = Text("Eve is detected")
        detection_probability = Text("Detection probability:")

        wrong_basis.to_edge(UP)
        wrong_bit.next_to(wrong_basis, 7 * DOWN)
        detected.next_to(wrong_bit, 7 * DOWN)
        detection_probability.to_edge(DOWN)

        arr1 = Arrow(
            wrong_basis.get_edge_center(DOWN), wrong_bit.get_edge_center(UP)
        )
        arr2 = Arrow(
            wrong_bit.get_edge_center(DOWN), detected.get_edge_center(UP)
        )

        p1 = MathTex(r"p=\frac 1 2")
        p2 = MathTex(r"p=\frac 1 2")
        p3 = MathTex(r"\frac{1}{2^2} = \frac 1 4")

        p1.next_to(arr1, RIGHT)
        p2.next_to(arr2, RIGHT)
        p3.next_to(detection_probability, RIGHT)

        self.play(FadeIn(wrong_basis))
        self.wait()
        self.play(FadeIn(wrong_bit), Write(arr1))
        self.wait()
        self.play(FadeIn(detected), Write(arr2))
        self.wait()
        self.play(Write(p1))
        self.play(Write(p2))
        self.wait()
        self.play(FadeIn(detection_probability))
        self.play(Write(p3))
        self.wait(3)
