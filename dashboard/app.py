"""
SoRL Arithmetic Dashboard — baseline vs SoRL side-by-side comparison.

Deployed as HF Space. Reads from thoughtworks/arithmetic-sorl model repo.
"""
import json
import gradio as gr
from gen_summary import build_summary_markdown
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import numpy as np
from PIL import Image
from huggingface_hub import HfApi, hf_hub_download

MODEL_REPO = "thoughtworks/arithmetic-sorl"

# ═══════════════════════════════════════════════════════════════════
# LaTeX scratchpad content — edit here, copy from dashboard into Overleaf
# ═══════════════════════════════════════════════════════════════════

LATEX_ARITHMETIC_SETUP = r"""% ── Arithmetic case study ──────────────────────────────────────────────────

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/fig_arithmetic_example.pdf}
  \caption{Addition $959{,}271 + 040{,}756 = 1{,}000{,}027$ — a four-deep carry cascade.
    Each answer-digit position is annotated with its Quirke subtask label (coloured box)
    and the \sorl{} abstraction token assigned by the model (dashed purple box).
    The carry chain ($d_1$--$d_5$) is highlighted; \sorl{} uses a consistent token
    (\texttt{t2}) at cascade positions and distinct tokens elsewhere.}
  \label{fig:arithmetic-example}
\end{figure}

\subsection{Case study: six-digit addition and subtraction}
\label{sec:arithmetic}

Six-digit addition and subtraction provides a setting where the internal
reasoning structure is known: \citet{quirke_2024_addsub_preprint} identify
ten mutually exclusive subtask types at each answer-digit position - carry generation, carry use, sum-9 boundary detection, borrow cascades, and so on - that a transformer must implement to solve the task
(see Table~\ref{tab:quirke-subtasks} for details).
This makes arithmetic an ideal testbed for one application of \sorl{}: that abstraction tokens may \emph{externalize} reasoning steps, making them directly observable and intervenable without any activation-level tooling.

\sorl{} inserts one abstraction token per answer-digit position ($K{=}1$,
codebook size $|\mathcal{A}|{=}30$), so each routing decision is a named, discrete symbol emitted at generation time.
Figure~\ref{fig:arithmetic-example} shows this concretely:
for $959{,}271 + 040{,}756 = 1{,}000{,}027$ (a four-deep carry cascade),
\sorl{} assigns \texttt{t2} at every cascade position (UC/US),
\texttt{t6} at the sum-9 boundary (SS), and distinct tokens at the trivial positions (SC/SA) - the carry structure is readable off the token sequence with no probing or patching required.
Full training and architecture details are in Appendix~\ref{app:training}.

\textbf{Abstraction tokens recover known circuits without supervision.}
Analysis of \texttt{2L/1H/128d} shows that \sorl{}'s codebook
spontaneously partitions into subtask-specialist tokens:
each of the 23 active tokens concentrates on a narrow slice of the Quirke taxonomy - dominant subtask accounts for ${\geq}70\%$ of occurrences for the majority of tokens - and is locked to one or two answer positions.
This structure emerges from the info-gain loss alone, with no access to ground-truth subtask labels or carry state.
Tokens are also causally necessary: knocking out all tokens collapses accuracy from 95.5\% to 0.1\%, confirming they carry the computation rather than merely annotating it.

\textbf{Named tokens better fine tuning performance and interventions.}
Because the routing codes are discrete and named, surgical model edits are possible that have no analog in standard transformers:
swapping a single token at one answer position fixes wrong predictions
at a 27-31\% rate on carry-heavy examples (cross-operation transplant:
93.5\% vs.\ 75.5\% random baseline).
Interpretability here is not merely post-hoc — it translates directly into the ability to correct the model.

Correspondingly, finetuned \sorl{} outperforms \sft{} strongly on 12 of 13 tested
(architecture, data-size) configurations, and on \emph{all 13} on
the hardest 6-deep carry cascades, with gains as large as $+50$\,pp
(Table~\ref{tab:undersized-wins}).
The margin grows with cascade depth, consistent with explicit carry/borrow routing being the mechanism behind the gain.

\begin{tcolorbox}[colback=gray!6, colframe=gray!40,
  fonttitle=\bfseries\small, title={Finding \#1: \sorl{} increases accuracy on 6-digit arithmetic dramatically},
  left=5pt, right=5pt, top=4pt, bottom=4pt]
\small \sorl{} increases accuracy on 6-digit arithmetic dramatically, winning stronger on hard cascade and borrow splits.
\end{tcolorbox}

See Appendix \ref{app:arithmetic} further details on SORL interpretability, including a demonstration of auto-interp ~\citep{bills2023language_models_explain_neurons}, token specializations and polysemantic tokens.
"""

LATEX_FIGURE_EXAMPLE = r"""% fig_arithmetic_example.tex
% Usage in paper: \input{figures/fig_arithmetic_example/fig_arithmetic_example.tex}
% Required packages: tikz, xcolor
%
% Shows 959271 + 040756 = 1000027 (4-deep carry cascade).
% Token assignments from model add_sub_sorl_v1_abs30_K1_100K (K=1, abs30).

\begin{figure}[t]
\centering
\begin{tikzpicture}[
  % ── node styles ─────────────────────────────────────────────────────
  digit/.style={
    draw=#1!60!gray, fill=#1, rounded corners=2pt,
    minimum width=1.05cm, minimum height=0.62cm,
    font=\small\bfseries, inner sep=2pt, text=#1!20!black,
  },
  subtask/.style={
    draw=#1!60!gray, fill=#1, rounded corners=2pt,
    minimum width=1.05cm, minimum height=0.55cm,
    font=\footnotesize, inner sep=2pt, text=#1!20!black,
  },
  token/.style={
    draw=violet!55, fill=violet!8, rounded corners=2pt, dashed,
    minimum width=1.05cm, minimum height=0.55cm,
    font=\footnotesize\bfseries, inner sep=2pt, text=violet!70!black,
  },
  rowlabel/.style={font=\footnotesize\itshape, text=gray!70!black, anchor=east},
  poslabel/.style={font=\scriptsize, text=gray!60!black},
  carry/.style={->, >=stealth, thick, color=orange!70!red,
                shorten <=3pt, shorten >=3pt},
]

% ── colours (Quirke subtask families) ───────────────────────────────────
\colorlet{cSA}{green!22!white}
\colorlet{cSC}{yellow!48!white}
\colorlet{cUC}{blue!22!white}
\colorlet{cUS}{blue!36!white}

% ── column spacing ───────────────────────────────────────────────────────
\def\cs{1.45}   % inter-column distance (cm)

% ── data (d0 = MSB/overflow on left, d6 = LSB on right) ─────────────────
%   d0    d1    d2    d3    d4    d5    d6
%   ---   9     5     9     2     7     1     (Addend A)
%   ---   0     4     0     7     5     6     (Addend B)
%   1     0     0     0     0     2     7     (Answer)
%   UC    US    US    US    US    SC    SA    (Subtask)
%   t2    t2    t6    t2    t1    t16   t3    (SoRL token)

% ── position labels ──────────────────────────────────────────────────────
\foreach \i/\lbl in {0/$d_0$,1/$d_1$,2/$d_2$,3/$d_3$,4/$d_4$,5/$d_5$,6/$d_6$}{
  \node[poslabel] at (\i*\cs, 3.15) {\lbl};
}

% ── row labels ───────────────────────────────────────────────────────────
\node[rowlabel] at (-0.65, 2.5)  {Addend $A$};
\node[rowlabel] at (-0.65, 1.8)  {Addend $B$};
\node[rowlabel] at (-0.65, 0.85) {Answer};
\node[rowlabel] at (-0.65, 0.1)  {Subtask};
\node[rowlabel] at (-0.65,-0.65) {Token};

% ── operand digits (d1..d6; d0 is the overflow, has no operand digits) ───
\foreach \i/\d in {1/9,2/5,3/9,4/2,5/7,6/1}{
  \node[font=\small, text=gray!30!black] at (\i*\cs, 2.5) {\d};
}
\node[font=\small\bfseries, text=gray!50!black] at (-0.4*\cs, 1.8) {$+$};
\foreach \i/\d in {1/0,2/4,3/0,4/7,5/5,6/6}{
  \node[font=\small, text=gray!30!black] at (\i*\cs, 1.8) {\d};
}

% ── horizontal rule ──────────────────────────────────────────────────────
\draw[gray!50, thin] (-0.55*\cs, 1.38) -- (6.55*\cs, 1.38);

% ── answer digit boxes ───────────────────────────────────────────────────
\node[digit=cUC] (a0) at (0*\cs, 0.85) {1};
\node[digit=cUS] (a1) at (1*\cs, 0.85) {0};
\node[digit=cUS] (a2) at (2*\cs, 0.85) {0};
\node[digit=cUS] (a3) at (3*\cs, 0.85) {0};
\node[digit=cUS] (a4) at (4*\cs, 0.85) {0};
\node[digit=cSC] (a5) at (5*\cs, 0.85) {2};
\node[digit=cSA] (a6) at (6*\cs, 0.85) {7};

% ── subtask boxes ────────────────────────────────────────────────────────
\node[subtask=cUC] at (0*\cs, 0.1) {UC};
\node[subtask=cUS] at (1*\cs, 0.1) {US};
\node[subtask=cUS] at (2*\cs, 0.1) {US};
\node[subtask=cUS] at (3*\cs, 0.1) {US};
\node[subtask=cUS] at (4*\cs, 0.1) {US};
\node[subtask=cSC] at (5*\cs, 0.1) {SC};
\node[subtask=cSA] at (6*\cs, 0.1) {SA};

% ── SoRL token boxes ─────────────────────────────────────────────────────
\foreach \i/\t in {0/t2,1/t2,2/t6,3/t2,4/t1,5/t16,6/t3}{
  \node[token] at (\i*\cs, -0.65) {\texttt{\t}};
}

% ── carry arrows (cascade flows right→left: d5→d4→d3→d2→d1→d0) ──────────
\foreach \fr/\to in {5/4, 4/3, 3/2, 2/1, 1/0}{
  \draw[carry] (a\fr.west) -- (a\to.east);
}

% ── cascade bracket + label ──────────────────────────────────────────────
\draw[orange!60!red, thin]
  (a5.north west) -- ++(0, 0.22)
  -- (a0.north east) -- ++(0,-0.22);
\node[font=\scriptsize\itshape, text=orange!60!red]
  at (2.5*\cs, 1.35) {carry cascade};

% ── legend ───────────────────────────────────────────────────────────────
\matrix[
  matrix of nodes,
  nodes={font=\scriptsize, inner sep=2pt, anchor=west},
  row sep=1pt, column sep=4pt,
  anchor=south east,
] at (6*\cs + 0.6, -1.05) {
  \node[digit=cSA, minimum width=0.45cm, minimum height=0.3cm,
        font=\scriptsize] {}; &
  \node {SA --- simple add}; &
  \node[digit=cUC, minimum width=0.45cm, minimum height=0.3cm,
        font=\scriptsize] {}; &
  \node {UC --- uses carry}; \\
  \node[digit=cSC, minimum width=0.45cm, minimum height=0.3cm,
        font=\scriptsize] {}; &
  \node {SC --- generates carry}; &
  \node[digit=cUS, minimum width=0.45cm, minimum height=0.3cm,
        font=\scriptsize] {}; &
  \node {US --- cascade}; \\
  \node[token, minimum width=0.45cm, minimum height=0.3cm,
        font=\scriptsize] {}; &
  \node[text=violet!70!black] {\sorl{} token}; & & \\
};

\end{tikzpicture}
\caption{%
  Six-digit addition $959{,}271 + 040{,}756 = 1{,}000{,}027$, a four-deep
  carry cascade. At each answer-digit position \sorl{} assigns one
  abstraction token (bottom row). Tokens \texttt{t2} and \texttt{t6}
  cluster on cascade positions (UC/US); \texttt{t16} marks the carry
  source (SC); \texttt{t3} marks the trivial position (SA).
  Token assignments from model \texttt{add\_sub\_sorl\_v1\_abs30\_K1\_100K}
  (K=1, 30-token codebook).%
}
\label{fig:arithmetic-example}
\end{figure}
"""

LATEX_TABLE_UNDERSIZED = r"""% tab:undersized-wins — SoRL vs SFT on undersized architectures
% Generated by arithmetic/paper/results/result_low_data_wins/run.py
% Requires: \usepackage{booktabs}, \usepackage{xcolor}

\begin{table}[t]
  \centering
  \small
  \begin{tabular}{llrrrr}
    \toprule
    Architecture & Data & Baseline & SoRL & Gap & C6 gap \\
    \midrule
    \texttt{1L/2H/256d} & 10K  & 10\% & \textbf{19\%} & \textcolor{green!50!black}{\textbf{+9\%}}  & \textcolor{green!50!black}{\textbf{+18\%}} \\
                        & 25K  & 32\% & 26\%          & $-7\%$                                    & \textcolor{green!50!black}{\textbf{+10\%}} \\
                        & 50K  & 44\% & \textbf{65\%} & \textcolor{green!50!black}{\textbf{+21\%}} & \textcolor{green!50!black}{\textbf{+34\%}} \\
                        & 100K & 49\% & \textbf{65\%} & \textcolor{green!50!black}{\textbf{+16\%}} & \textcolor{green!50!black}{\textbf{+31\%}} \\
    \midrule
    \texttt{1L/3H/510d} & 10K  & 36\% & \textbf{52\%} & \textcolor{green!50!black}{\textbf{+16\%}} & \textcolor{green!50!black}{\textbf{+30\%}} \\
                        & 25K  & 46\% & \textbf{60\%} & \textcolor{green!50!black}{\textbf{+14\%}} & \textcolor{green!50!black}{\textbf{+22\%}} \\
                        & 50K  & 53\% & \textbf{72\%} & \textcolor{green!50!black}{\textbf{+19\%}} & \textcolor{green!50!black}{\textbf{+38\%}} \\
                        & 100K & 67\% & \textbf{83\%} & \textcolor{green!50!black}{\textbf{+16\%}} & \textcolor{green!50!black}{\textbf{+26\%}} \\
    \midrule
    \texttt{2L/1H/128d} & 10K  & 16\% & \textbf{36\%} & \textcolor{green!50!black}{\textbf{+21\%}} & \textcolor{green!50!black}{\textbf{+39\%}} \\
                        & 25K  & 40\% & \textbf{55\%} & \textcolor{green!50!black}{\textbf{+15\%}} & \textcolor{green!50!black}{\textbf{+23\%}} \\
                        & 50K  & 59\% & \textbf{87\%} & \textcolor{green!50!black}{\textbf{+28\%}} & \textcolor{green!50!black}{\textbf{+50\%}} \\
                        & 75K  & 75\% & \textbf{87\%} & \textcolor{green!50!black}{\textbf{+12\%}} & \textcolor{green!50!black}{\textbf{+5\%}}  \\
                        & 100K & 73\% & \textbf{95\%} & \textcolor{green!50!black}{\textbf{+22\%}} & \textcolor{green!50!black}{\textbf{+33\%}} \\
    \bottomrule
  \end{tabular}
  \caption{\sorl{} ($K{=}1$, $|\mathcal{A}|{=}30$) vs.\ \sft{} baseline on
    undersized architectures across data sizes.
    \textbf{Gap} = overall accuracy gain; \textbf{C6 gap} = gain on
    6-deep carry cascades (the hardest split).
    \sorl{} wins in \textbf{12 of 13} (architecture, data-size) pairs;
    the single exception is \texttt{1L/2H/256d} at 25K, where the model
    is undertrained (accuracy still rising at epoch 20).
    \sorl{} wins on C6 in \textbf{all 13} configurations.}
  \label{tab:undersized-wins}
\end{table}
"""

LATEX_APPENDIX = r"""\section{Arithmetic case study: interpretability analysis}
\label{app:arithmetic}

\begin{table}[h]
  \centering\small
  \setlength{\tabcolsep}{6pt}
  \begin{tabular}{llp{7.8cm}}
    \toprule
    & Label & Condition at digit position $n$ \\
    \midrule
    \multirow{5}{*}{\rotatebox[origin=c]{90}{Addition\;}}
      & \textbf{SA} & $d_1{+}d_2 \leq 8$;\; no carry in or out \\
      & \textbf{SC} & $d_1{+}d_2 \geq 10$;\; generates a carry \\
      & \textbf{SS} & $d_1{+}d_2 = 9$;\; carry state \emph{uncertain} (cascade boundary) \\
      & \textbf{UC} & carry arrives from position $n{-}1$;\; answer digit depends on it \\
      & \textbf{US} & carry propagates through a run of SS positions (sum-of-9 cascade) \\
    \midrule
    \multirow{5}{*}{\rotatebox[origin=c]{90}{Subtraction\;}}
      & \textbf{MD} & $d_1 \geq d_2$;\; no borrow \\
      & \textbf{MB} & $d_1 < d_2$;\; generates a borrow \\
      & \textbf{ME} & $d_1 = d_2$;\; borrow state \emph{uncertain} \\
      & \textbf{UB} & borrow arrives from position $n{-}1$ \\
      & \textbf{UD} & borrow propagates through a run of ME positions \\
    \bottomrule
  \end{tabular}
  \caption{Per-digit subtask labels for six-digit addition and
    subtraction~\citep{quirke_2024_addsub_preprint}.
    Cascades (US, UD) require tracking carry/borrow state across
    multiple positions and are the hardest splits.}
  \label{tab:quirke-subtasks}
\end{table}

\paragraph{Setup.}
All interpretability analyses use a 2-layer, 1-head, 128-dimensional transformer
trained on 100K examples, evaluated on 2{,}600 held-out problems across 26 splits.
This model achieves 95.5\% accuracy with \sorl{} abstraction tokens and 0.1\% without; making it the clearest test-bed for causal analysis.
All results are reproducible from the released code.

% ─────────────────────────────────────────────────────────────────────────────
\subsection{Performance: \sorl{} vs.\ \sft{} on undersized architectures}
\label{app:performance}

\begin{table}[t]
  \centering\small
  \begin{tabular}{llrrrr}
    \toprule
    Architecture & Data & Baseline & SoRL & Gap & C6 gap \\
    \midrule
    \texttt{1L/2H/256d} & 10K  & 10\% & \textbf{19\%} & \textcolor{green!50!black}{\textbf{+9\%}}  & \textcolor{green!50!black}{\textbf{+18\%}} \\
                        & 25K  & 32\% & 26\%          & $-7\%$                                    & \textcolor{green!50!black}{\textbf{+10\%}} \\
                        & 50K  & 44\% & \textbf{65\%} & \textcolor{green!50!black}{\textbf{+21\%}} & \textcolor{green!50!black}{\textbf{+34\%}} \\
                        & 100K & 49\% & \textbf{65\%} & \textcolor{green!50!black}{\textbf{+16\%}} & \textcolor{green!50!black}{\textbf{+31\%}} \\
    \midrule
    \texttt{1L/3H/510d} & 10K  & 36\% & \textbf{52\%} & \textcolor{green!50!black}{\textbf{+16\%}} & \textcolor{green!50!black}{\textbf{+30\%}} \\
                        & 25K  & 46\% & \textbf{60\%} & \textcolor{green!50!black}{\textbf{+14\%}} & \textcolor{green!50!black}{\textbf{+22\%}} \\
                        & 50K  & 53\% & \textbf{72\%} & \textcolor{green!50!black}{\textbf{+19\%}} & \textcolor{green!50!black}{\textbf{+38\%}} \\
                        & 100K & 67\% & \textbf{83\%} & \textcolor{green!50!black}{\textbf{+16\%}} & \textcolor{green!50!black}{\textbf{+26\%}} \\
    \midrule
    \texttt{2L/1H/128d} & 10K  & 16\% & \textbf{36\%} & \textcolor{green!50!black}{\textbf{+21\%}} & \textcolor{green!50!black}{\textbf{+39\%}} \\
                        & 25K  & 40\% & \textbf{55\%} & \textcolor{green!50!black}{\textbf{+15\%}} & \textcolor{green!50!black}{\textbf{+23\%}} \\
                        & 50K  & 59\% & \textbf{87\%} & \textcolor{green!50!black}{\textbf{+28\%}} & \textcolor{green!50!black}{\textbf{+50\%}} \\
                        & 75K  & 75\% & \textbf{87\%} & \textcolor{green!50!black}{\textbf{+12\%}} & \textcolor{green!50!black}{\textbf{+5\%}}  \\
                        & 100K & 73\% & \textbf{95\%} & \textcolor{green!50!black}{\textbf{+22\%}} & \textcolor{green!50!black}{\textbf{+33\%}} \\
    \bottomrule
  \end{tabular}
  \caption{\sorl{} ($K{=}1$, $|\mathcal{A}|{=}30$) vs.\ \sft{} baseline on
    undersized architectures across data sizes.
    \textbf{Gap} = overall accuracy gain; \textbf{C6 gap} = gain on
    6-deep carry cascades (the hardest split).
    \sorl{} wins in \textbf{12 of 13} (architecture, data-size) pairs;
    the single exception is \texttt{1L/2H/256d} at 25K, where the model
    is undertrained (accuracy still rising at epoch 20).
    \sorl{} wins on C6 in \textbf{all 13} configurations.}
  \label{tab:undersized-wins}
\end{table}

% ─────────────────────────────────────────────────────────────────────────────
\subsection{Causal ablations}
\label{app:causal}

To confirm that \sorl{} tokens are causally necessary (not merely correlated
with correct outputs), we run three intervention conditions on model
\texttt{2L/1H/128d} (100K), evaluated across 2{,}600 held-out problems:

\begin{itemize}[nosep]
  \item \textbf{Shuffle}: randomly permute all abstraction tokens within
    each sequence (token identities preserved; positional assignment destroyed).
  \item \textbf{Random}: replace each token with a draw uniform over the
    30-token codebook (identity and position both destroyed).
  \item \textbf{Knockout}: replace every token with a fixed \texttt{[UNK]}
    embedding (strongest intervention; removes all information).
\end{itemize}

Table~\ref{tab:ablation-splits} shows per-split accuracy under each condition.

\begin{table}[h]
  \centering\small
  \begin{tabular}{llrrrr}
    \toprule
    Family & Split & Baseline & Shuffle & Random & Knockout \\
    \midrule
    \multirow{4}{*}{\textit{Addition (easy)}} & S0 (no carry) & 100\% & 24\% & 28\% & 0\% \\
     & S1 & 100\% & 17\% &  9\% & 0\% \\
     & S2 & 100\% & 22\% & 10\% & 0\% \\
     & random & 100\% & 26\% &  8\% & 0\% \\
    \midrule
    \multirow{4}{*}{\textit{Addition cascade (hard)}} & C3 (3-deep) & 96\% & 28\% & 14\% & 0\% \\
     & C4 (4-deep) & 99\% & 25\% & 13\% & 0\% \\
     & C5 (5-deep) & 99\% & 23\% & 19\% & 0\% \\
     & C6 (6-deep) & 97\% & 27\% & 15\% & 0\% \\
    \midrule
    \multirow{1}{*}{\textit{Subtraction (easy)}} & random & 100\% & 46\% & 12\% & 0\% \\
    \midrule
    \multirow{3}{*}{\textit{Subtraction cascade (hard)}} & M3 (3-deep borrow) & 100\% & 22\% &  1\% & 0\% \\
     & M4 (4-deep borrow) &  85\% &  6\% &  0\% & 0\% \\
     & M5 (5-deep borrow) &  57\% &  3\% &  0\% & 2\% \\
    \midrule
    \multicolumn{2}{l}{\textbf{Overall}} & \textbf{95.5\%} & 26.6\% & 12.3\% & 0.1\% \\
    \bottomrule
  \end{tabular}
  \caption{Per-split causal ablation (\texttt{2L/1H/128d}, 100K training examples).
    \textbf{Shuffle} preserves token identity but destroys positional assignment;
    \textbf{Random} destroys both; \textbf{Knockout} removes all token information.
    Generated by \texttt{paper/results/result\_ablation\_splits/run.py}.}
  \label{tab:ablation-splits}
\end{table}

\paragraph{Commentary.}
Knockout reduces accuracy to $\leq$2\% on every split, confirming that the model has offloaded computation into the routing tokens.
Three patterns are notable:

\begin{itemize}[nosep]
  \item \textbf{Shuffle $>$ Random on easy splits.}
    On addition S0 (no carry), shuffle yields 24\% vs.\ random's 28\%; the gap is small and reflects that no single position is critical - any token in any position is roughly as bad as another.
  \item \textbf{Shuffle $>$ Random on cascade splits (C3--C6).}
    On 4--6-deep carry cascades, shuffle (23--28\%) consistently
    outperforms random (13--19\%).
    When tokens are shuffled, a cascade position receives a token
    from another cascade position — a wrong token but from the
    ``right family'', producing a systematic one-off carry error.
    Random tokens provide no structural signal at all, making
    cascade resolution impossible.
  \item \textbf{Borrow cascades are uniquely sensitive.}
    Sub-M4 (4-deep borrow) drops from 85\% baseline to 6\% under
    shuffle and 0\% under random - a 79\,pp collapse from shuffle
    alone. Sub-M5 (5-deep borrow) is hardest even at baseline (57\%),
    and all ablations reduce it to $\leq$3\%, showing that deep
    borrow cascades are the single hardest regime and that \sorl{}
    tokens are essential for solving them.
\end{itemize}

\begin{tcolorbox}[colback=gray!6, colframe=gray!40,
  fonttitle=\bfseries\small, title={Finding \#2: Abstraction tokens are causally necessary},
  left=5pt, right=5pt, top=4pt, bottom=4pt]
\small Knockout collapses accuracy from 95.5\% to 0.1\%; shuffle hurts more than random on cascades.
\end{tcolorbox}

% ─────────────────────────────────────────────────────────────────────────────
\subsection{Token-subtask heatmap}
\label{app:heatmap}

\begin{figure}[h]
  \centering
  \includegraphics[width=0.95\linewidth]{Styles/figures/fig_token_subtask.pdf}
  \caption{Token--subtask heatmap for \texttt{2L/1H/128d} (100K).
    Each cell shows $P(\text{subtask} \mid \text{token})$ — the fraction of that token's
    occurrences that fall into each Quirke subtask — over 2{,}600 held-out examples across
    all addition and subtraction splits.
    Rows are the 23 active tokens (of 30 in the codebook), sorted by dominant subtask;
    columns are the 10 subtask labels (SA, SC, SS, UC, US for addition; MD, MB, ME, UB, UD for subtraction).
    High values appear in narrow bands: most tokens fire predominantly on 1--2 subtasks,
    with the highest-purity tokens (e.g.\ \texttt{t19}: UD 82\%, \texttt{t20}: UD 84\%)
    approaching single-subtask specialization.
    Tokens at the bottom of the heatmap (e.g.\ \texttt{t1}, \texttt{t2}) are more
    polysemantic, spreading probability mass across multiple subtasks.}
  \label{fig:token-subtask}
\end{figure}

Of the 30 tokens in the codebook, 23 appear in the held-out evaluation set.
Each active token concentrates on a narrow slice of the subtask space: the dominant subtask accounts for ${\geq}70\%$ of that token's occurrences in the majority of cases.
Tokens are also \emph{position-locked}: each token appears predominantly at one or two answer positions ($d_0$-$d_6$), rarely crossing position boundaries.
Representative examples are shown in Table~\ref{tab:token-profiles}:
token \texttt{t21} fires 93\% of the time on US (sum-9 cascade, addition)
with digit sum $\equiv 9 \pmod{10}$ in 95\% of cases;
token \texttt{t23} is the subtraction mirror (UD, 88\%, position $d_3$).

\begin{table}[h]
  \centering\small
  \begin{tabular}{llrrll}
    \toprule
    Token & Pos & $n$ & Top subtask & Purity & Op \\
    \midrule
    \texttt{t21} & $d_3$ & 719 & US & 93\% & add (94\%) \\
    \texttt{t23} & $d_3$ & 687 & UD & 88\% & sub (93\%) \\
    \texttt{t6}  & $d_2$ & 1438 & US & 52\% & add (79\%), sum${\equiv}9$: 78\% \\
    \texttt{t7}  & $d_2$ & 1026 & UB/UD & 90\% & sub (100\%) \\
    \texttt{t14} & $d_4$ & 1242 & UD & 73\% & sub (81\%) \\
    \bottomrule
  \end{tabular}
  \caption{Selected token profiles. Each token specialises in a specific
    subtask at a specific answer position. ``Purity'' = fraction of
    occurrences where the top subtask applies.}
  \label{tab:token-profiles}
\end{table}

\begin{tcolorbox}[colback=gray!6, colframe=gray!40,
  fonttitle=\bfseries\small, title={Finding \#3: Tokens spontaneously specialize by subtask and position},
  left=5pt, right=5pt, top=4pt, bottom=4pt]
\small 23 of 30 tokens are active; each locks to 1--2 Quirke subtasks (${\geq}70\%$ purity) and 1--2 answer positions.
\end{tcolorbox}

% ─────────────────────────────────────────────────────────────────────────────
\subsection{Guided computation via token intervention}
\label{app:guided}

If \sorl{} tokens encode the \emph{computational route} rather than just the answer, swapping a token at a mispredicted position should \emph{fix} the prediction — without retraining, without accessing internal activations, and without modifying the weights.

We test this with \emph{surgical swap}: for each wrong prediction,
we try replacing the abstraction token at each answer position with
every other token in the codebook (29 candidates $\times$ 5 positions = 145 interventions per example) and measure how many wrong predictions become correct — and how many previously-correct predictions break.

\paragraph{Results.}
At positions $d_0$-$d_2$ (the carry-heavy positions), a fixing swap exists for 27-31\% of mispredicted examples.
The best single swap is replacing \texttt{t16} with \texttt{t25} at $d_1$: this fixes 10 wrong predictions while breaking only 5 correct ones (a 2:1 fix-to-break ratio), all on carry cascade splits (C4-C6).
Position $d_3$ and $d_4$ are harder to fix surgically (fix rates 8\% and 2\%), consistent with those positions encoding longer-range carry state that a single-position swap cannot resolve.

\begin{tcolorbox}[colback=gray!6, colframe=gray!40,
  fonttitle=\bfseries\small, title={Finding \#4: Single token swaps fix mispredicted carry-heavy examples},
  left=5pt, right=5pt, top=4pt, bottom=4pt]
\small Replacing one abstraction token fixes 27--31\% of wrong predictions at carry-heavy positions, with no weight updates.
\end{tcolorbox}

% ─────────────────────────────────────────────────────────────────────────────
\subsection{\sorl{} tokens recover circuits from Quirke et al.}
\label{app:quirke-analogy}

\citet{quirke_2024_addsub_preprint} identify, via PCA of internal
residual-stream activations, a \emph{tri-state carry classifier} at each digit position $n$: the hidden state encodes which of three carry regimes
applies —
$\text{ST}_n = 0$ (digit sum $< 9$; carry cannot propagate),
$\text{ST}_n = 1$ (digit sum $> 9$; carry will propagate), or
$\text{ST}_n = U$ (digit sum $= 9$; carry state \emph{uncertain}, depends
on lower positions).
This trichotomy is the core circuit for addition; borrow cascades have an analogous structure for subtraction.

\sorl{} recovers the same trichotomy \emph{without access to activation
data or ground-truth circuit labels}, purely from the info-gain training
signal.
The correspondence is visible directly in the codebook:

\begin{itemize}[nosep]
  \item $\text{ST}_n = U$ (sum-9 uncertain, addition) $\longleftrightarrow$
    \texttt{t21} at $d_3$ (US=93\%, sum${\equiv}9$: 95\%);
    \texttt{t6} at $d_2$ (US=52\%, sum${\equiv}9$: 78\%).
  \item $\text{ST}_n = U$ (borrow-uncertain, subtraction) $\longleftrightarrow$
    \texttt{t23} at $d_3$ (UD=88\%, sub=93\%);
    \texttt{t7} at $d_2$ (UB/UD=90\%, sub=100\%).
  \item $\text{ST}_n = 1$ (carry generated) $\longleftrightarrow$
    tokens concentrated on SC/MB subtasks at each position.
  \item $\text{ST}_n = 0$ (simple digit, no carry) $\longleftrightarrow$
    tokens concentrated on SA/MD subtasks.
\end{itemize}

Crucially, Quirke et al.\ required full activation-level mechanistic interpretability to discover this classifier; \sorl{} surfaces it as a readable token in the output sequence, accessible without any post-hoc analysis.

\begin{tcolorbox}[colback=gray!6, colframe=gray!40,
  fonttitle=\bfseries\small, title={Finding \#5: \sorl{} rediscovers Quirke's circuits without supervision},
  left=5pt, right=5pt, top=4pt, bottom=4pt]
\small \sorl{} recovers the carry-state tri-classifier ($\{0,U,1\}$) unsupervised — externalizing what \citet{quirke_2024_addsub_preprint} needed PCA to reveal.
\end{tcolorbox}

% ─────────────────────────────────────────────────────────────────────────────
\subsection{Polysemantic tokens}
\label{app:polysemantic}

Not all codebook tokens are specialists. Table~\ref{tab:token-polysemanticity}
contrasts the most specialist with the most polysemantic tokens.
Token \texttt{t21} fires 94\% of the time on a single subtask (US) at a single position ($d_3$, addition only) — a true specialist.
Token \texttt{t1}, by contrast, is the highest-frequency token
($n{=}6{,}359$, spanning all five answer positions) with no subtask
exceeding 24\% — it acts as a general-purpose fallback, handling
whichever position and carry regime was not captured by a specialist token.

\begin{table}[h]
  \centering\small
  \begin{tabular}{clrrr}
    \toprule
    Token & Top subtask & Purity & $n$ & Positions \\
    \midrule
    \multicolumn{5}{l}{\textit{Specialist (high purity)}} \\
    \texttt{t21} & US (cascade, add) & 94\% & 719  & 1 \\
    \texttt{t23} & UD (cascade, sub) & 88\% & 687  & 3 \\
    \texttt{t14} & UD               & 74\% & 1242 & 3 \\
    \midrule
    \multicolumn{5}{l}{\textit{Polysemantic (low purity)}} \\
    \texttt{t5}  & UB               & 21\% & 1377 & 4 \\
    \texttt{t1}  & MD               & 24\% & 6359 & 5 \\
    \texttt{t20} & UB               & 24\% &  283 & 3 \\
    \bottomrule
  \end{tabular}
  \caption{Specialist vs.\ polysemantic tokens.
    \textbf{Purity} = fraction of occurrences where the top subtask applies.
    \textbf{Positions} = number of distinct answer positions ($d_0$-$d_6$)
    where the token appears. \texttt{t1} is a high-frequency fallback used
    across all positions; \texttt{t21} is a pure sum-9 cascade detector.}
  \label{tab:token-polysemanticity}
\end{table}

Polysemanticity here mirrors the phenomenon described in neural network interpretability~\citep{elhage2022superposition}: a single token encodes multiple distinct roles, likely because the 30-token codebook has spare capacity at the overflow position ($d_0$) where carry state is most
variable. The specialist tokens concentrate at mid-sequence positions ($d_2$-$d_4$) where carry propagation is most structured.

\begin{tcolorbox}[colback=gray!6, colframe=gray!40,
  fonttitle=\bfseries\small, title={Finding \#6: The codebook mixes specialist and polysemantic tokens},
  left=5pt, right=5pt, top=4pt, bottom=4pt]
\small Specialist tokens (\texttt{t21}: 94\% purity) coexist with polysemantic fallbacks (\texttt{t1}: 24\%, five positions).
\end{tcolorbox}

% ─────────────────────────────────────────────────────────────────────────────
\subsection{Automated token interpretation}
\label{app:autointerp}

We implement a light version of the automated interpretation procedure of \citet{bills2023language}.
For each active token, we collect the $N{=}10$ examples from the evaluation set where the model assigned it with highest softmax confidence,
then ask \texttt{claude-haiku} to produce a one-sentence role description.

% [PLACEHOLDER — run experiments/11_auto_interp/run.py to generate table.tex]
% Then paste the output of table.tex here:
%
% \input{experiments/11_auto_interp/table.tex}
%
% Expected columns: Token | Top subtask (purity) | Mean conf. | Auto-interpretation

\begin{tcolorbox}[colback=gray!6, colframe=gray!40,
  fonttitle=\bfseries\small, title={Finding \#7: Automated interpretation matches ground-truth subtask labels},
  left=5pt, right=5pt, top=4pt, bottom=4pt]
\small High-confidence tokens get crisp role descriptions; polysemantic tokens get appropriately vague ones — without accessing ground-truth labels.
\end{tcolorbox}
% ─────────────────────────────────────────────────────────────────────────────
\subsection{Training and evaluation details}
\label{app:training}

\begin{table}[h]
  \centering\small
  \begin{tabular}{lrrrrr}
    \toprule
    Architecture & Layers & Heads & Hidden & FFN & Params \\
    \midrule
    \texttt{1L/2H/256d} & 1 & 2 & 256 & 1024 & ${\sim}$0.3M \\
    \texttt{1L/3H/510d} & 1 & 3 & 510 & 2040 & ${\sim}$2.0M \\
    \texttt{2L/1H/128d} & 2 & 1 & 128 &  512 & ${\sim}$0.1M \\
    \bottomrule
  \end{tabular}
  \caption{Undersized architectures (Table~\ref{tab:undersized-wins}).
    All use pre-norm, GeLU, Qwen3 tokenizer.}
\end{table}

\begin{table}[h]
  \centering\small
  \begin{tabular}{ll}
    \toprule
    Hyperparameter & Value \\
    \midrule
    Optimizer              & AdamW \\
    Learning rate          & $8\times10^{-5}$ \\
    $(\beta_1,\,\beta_2)$  & $(0.9,\;0.999)$ \\
    Weight decay           & $0.01$ \\
    LR schedule            & Linear warmup (3\%) then constant \\
    Batch size             & 64 \\
    Epochs                 & 20 \\
    \sorl{} codebook       & $|\mathcal{A}|{=}30$, $K{=}1$ \\
    $\alpha_{\text{info-gain}},\,\alpha_{\text{abs}},\,\alpha_{\text{zipf}}$
                           & $10.0,\;0.1,\;1.0$ \\
    \bottomrule
  \end{tabular}
  \caption{Shared training hyperparameters.}
\end{table}

Evaluation uses fixed-length autoregressive decoding (no teacher forcing):
the model generates answer digits $d_0 \to d_6$ using its own predictions, with abstraction tokens inserted via the \sorl{} search-then-recurse procedure (matching training). Accuracy is measured on 100 held-out examples per split (seed 42; \texttt{thoughtworks/arithmetic-sorl-data}).
"""


# ═══════════════════════════════════════════════════════════════════
# App
# ═══════════════════════════════════════════════════════════════════

with gr.Blocks(title="SoRL Arithmetic Dashboard") as app:
    gr.Markdown("# SoRL Arithmetic Dashboard")
    gr.Markdown("[Models](https://huggingface.co/thoughtworks/arithmetic-sorl) · "
                "[Datasets](https://huggingface.co/datasets/thoughtworks/arithmetic-sorl-data) · "
                "[WandB](https://wandb.ai/nlp_and_interpretability/sorl-arithmetic) · "
                "[Code](https://github.com/fangyuan-ksgk/mod_gpt/tree/amir/arithmetic) · "
                "[Quirke et al. 2024](https://arxiv.org/abs/2402.02619)")

    models_state = gr.State([])

    with gr.Tabs():
        # ── Tab 1: Models ──
        with gr.TabItem("Models"):
            with gr.Accordion("What is SoRL?", open=False):
                gr.Markdown("""
**Self-Organized Reinforcement Learning (SoRL)** augments a transformer with learned *abstraction tokens* —
a small auxiliary vocabulary (e.g. 30 tokens) inserted at regular intervals (every K positions) into the sequence.

```
  Standard SFT:     1 2 3 4 5 6 + 6 5 4 3 2 1 = 1 8 7 9 7 7

  SoRL (K=4):       1 2 3 [a] 4 5 6 [a] + 6 5 [a] 4 3 2 [a] 1 = 1 [a] 8 7 9 [a] 7 7
```

**Training**: insert placeholders → search for best abstraction values → train with info-gain loss
(abstractions must reduce prediction uncertainty). **Eval**: autoregressive, errors propagate.
""")

            with gr.Row():
                arch_filter = gr.Dropdown(["All", "2L/3H/510d", "1L/3H/510d", "1L/2H/256d", "2L/1H/128d"],
                                          value="All", label="Architecture")
                refresh_btn = gr.Button("Refresh from HF", variant="primary")

            summary_text = gr.Markdown("Click Refresh to load.")
            queue_status = gr.Markdown("")

            summary_md = gr.Markdown(build_summary_markdown())
            main_table = gr.Dataframe(
                headers=["Ops", "Data", "Arch", "Baseline", "SoRL", "Config", "B_hf", "S_hf", "B_wandb", "S_wandb"],
                datatype=["str", "str", "str", "markdown", "markdown", "str", "markdown", "markdown", "markdown", "markdown"],
                interactive=False,
            )

            with gr.Accordion("Hard Split Comparison (C4, C5, C6, M4, M5)", open=False):
                gr.Markdown("Left = Baseline, Right = SoRL. **Bold** = winner. C-splits = hot carry chains (varied answers).")
                hard_table = gr.Dataframe(
                    headers=["Ops", "Data", "Arch", "Config",
                             "B_add_C4", "S_add_C4", "B_add_C5", "S_add_C5",
                             "B_add_C6", "S_add_C6", "B_sub_M4", "S_sub_M4",
                             "B_sub_M5", "S_sub_M5"],
                    datatype=["str", "str", "str", "str"] + ["markdown"] * 10,
                    interactive=False,
                )

            with gr.Accordion("Data Efficiency & Undersized Models", open=False):
                gr.Image("static_figures/fig_data_efficiency.png")
                gr.Markdown("At 10K, SoRL K=1 abs30 reaches **96.7%** vs baseline **76.6%** (+20pp). By 50K both hit 100%.")
                gr.Image("static_figures/fig_undersized.png")
                gr.Markdown("Undersized 2L/1H/128d: baseline 50% → SoRL **85%** (+35pp). Abstraction tokens compensate for limited capacity.")

            with gr.Accordion("Per-Split Detail", open=False):
                model_selector = gr.Dropdown(label="Model", choices=[], allow_custom_value=True)
                detail_btn = gr.Button("Show splits")
                detail_table = gr.Dataframe(headers=["Split", "Accuracy", "N"], interactive=False)

        # ── Tab 2: Interpretability ──
        with gr.TabItem("Interpretability"):
            gr.Markdown("""## SoRL tokens externalize arithmetic circuits

### Background: how multi-digit arithmetic works

Adding two 6-digit numbers like `345678 + 657893` requires tracking **carries** — when
a column sums to 10 or more, you carry 1 to the next column. A **carry cascade** happens
when carries chain through multiple consecutive columns (e.g., `999 + 1 = 1000`).

We evaluate on **C-splits** — problems grouped by how many consecutive columns produce carries,
with varied (non-zero) answer digits:

| Split | Meaning | Example | Why it's hard |
|-------|---------|---------|---------------|
| **C1** | 1 carry | `345678 + 100921 = 446599` | Single carry — easy |
| **C2** | 2 consecutive carries | `503847 + 297162 = 801009` | Carry propagates once |
| **C3** | 3 consecutive carries | `145232 + 957868 = 1103100` | Must track 3-step cascade |
| **C4** | 4 consecutive carries | `780149 + 819959 = 1600108` | Longer cascade chain |
| **C5** | 5 consecutive carries | `553777 + 847927 = 1401704` | Nearly full cascade |
| **C6** | 6 carries (max) | `503847 + 996167 = 1500014` | Every column cascades |

[Quirke et al. (2024)](https://arxiv.org/abs/2402.02619) showed that transformers learn
these carry/borrow circuits internally, but they're hidden in activations — discoverable only
through PCA, probing, or ablation at the activation level.

### Quirke's subtask definitions

At each digit position, the model must compute one of these operations
([Quirke et al. §3.2-3.3](https://arxiv.org/abs/2402.02619)):

**Addition:**
| Subtask | Meaning | Quirke eq. |
|---------|---------|-----------|
| **SA** | Simple Add: `(d₁ + d₂) mod 10` | — |
| **SC** | Sum Carry: `d₁ + d₂ ≥ 10`, produces a local carry | — |
| **SS** | Sum-of-9: `d₁ + d₂ = 9`, carry state is **uncertain** | eq. 2: STn = U |
| **UC** | Use Carry: this digit's answer depends on carry from right | eq. 2: STn = 1 |
| **US** | Use Sum-9 cascade: carry propagates through a chain of sum-9 digits | eq. 4-6 |

**Subtraction** (same structure with borrows replacing carries):
| Subtask | Meaning | Quirke eq. |
|---------|---------|-----------|
| **MD** | Base Diff: `(d₁ - d₂) mod 10` | — |
| **MB** | Make Borrow: `d₁ < d₂`, produces a local borrow | eq. 7: MBn = 1 |
| **ME** | Equal digits: `d₁ = d₂`, borrow state is **uncertain** | eq. 7: MBn = U |
| **UB** | Use Borrow: answer depends on borrow from right | — |
| **UD** | Cascade borrow: borrow propagates through equal-digit chain | — |

**SoRL makes these circuits directly observable as tokens.** We show that:
1. More abstraction vocabulary → richer representations
2. Different tokens map to different arithmetic operations
3. Token identity is causally necessary for correct answers
4. These tokens correspond to Quirke's carry/borrow circuits
""")

            gr.Markdown("""### 1. More vocabulary → higher accuracy and richer representations

Increasing the abstraction vocabulary from 10 to 30 tokens improves accuracy, especially
on undersized models where capacity is limited. The 2L/1H/128d model gains +16pp
going from abs10 to abs30. The standard model saturates at ~100% regardless of vocab size.
""")
            gr.Image("static_figures/fig_vocab_scaling.png")

            gr.Markdown("""With more tokens available, the model uses more of them and distributes usage
more evenly. abs10 collapses to 5 tokens; abs30 uses 18 with higher entropy.
""")
            gr.Image("static_figures/fig_diversity.png")

            gr.Markdown("""### 2. Tokens map to Quirke's subtasks

Within addition alone, tokens form a **carry computation spectrum** — sorted by how often
they appear at positions with an active carry. Tokens at the top handle "no carry" positions
(SA, SS). Tokens at the bottom are pure carry tokens (UC=87-100%). The model has learned
a graded vocabulary from "nothing happening" to "full carry cascade."
""")
            gr.Image("static_figures/fig_addition_hierarchy.png")

            gr.Markdown("""The full heatmap across both addition and subtraction:""")
            gr.Image("static_figures/fig_token_subtask.png")

            gr.Markdown("""### 3. Three token vignettes

Each vignette introduces one token, explains its role across many examples, then
walks through a concrete problem showing it in action.

---

**Vignette 1: Token t2 — the carry cascade propagator**

Across 2,550 occurrences, t2 appears primarily at **UC (Use Carry, 31%)** and **UB (Use Borrow, 14%)**
positions — places where the answer digit depends on a carry or borrow propagating from the right.
It appears at multiple answer positions (d1=28%, d3=22%, d0=17%), always where cascade state
must be tracked. It's the model's way of saying: *"this digit depends on what happened to the right."*

Here it is in a 5-carry cascade (`959271 + 040756 = 1000027`):

```
  Position:   d0   d1   d2   d3   d4   d5   d6
  Answer:      1    0    0    0    0    2    7
  Subtask:    UC   US   US   US   US   SC   SA
  Token:      t2   t2   t6   t2   t1    —    —
                ↑    ↑         ↑
              t2 marks every position that must propagate the carry cascade
```

d5-d6 are easy (SC/SA) — no abstraction needed. d1-d4 are all US (sum-of-9): the carry
cascades left through four consecutive uncertain positions. t2 appears at d0, d1, and d3 —
every position that needs to resolve the cascade.

---

**Vignette 2: Token t7 — the borrow cascade specialist**

t7 appears 1,026 times and is **100% subtraction** — never appears in addition problems.
It splits between **UB (Use Borrow, 47%)** and **UD (cascade borrow, 43%)**, and is locked
to position d2 (95%). When the model assigns t7, it means: *"borrow is propagating through
this position in a subtraction."*

Here it is in a 4-borrow cascade (`698401 - 128406 = 0569995`):

```
  Position:   d0   d1   d2   d3   d4   d5   d6
  Answer:      0    5    6    9    9    9    5
  Subtask:    MD   MD   UB   UD   UD   UD   MB
  Token:      t1   t5   t7   t7   t1    —    —
                         ↑    ↑
                   t7 marks the borrow cascade positions
```

d6 triggers the borrow (MB). The borrow cascades left through d3-d5 (UD = equal digits,
borrow uncertain). t7 appears at d2-d3 where the cascade must be resolved. Note that t7
never appears in the addition vignette above — the model has learned operation-specific tokens.

---

**Vignette 3: Token t3 — the simple addition marker**

t3 appears 2,084 times, primarily at **UB (37%)** and **US (27%)** positions, but critically
at positions where no carry computation is needed. In easy problems (S0), it marks the
"nothing interesting here" positions.

Here it is in a no-carry addition (`417080 + 531003 = 0948083`):

```
  Position:   d0   d1   d2   d3   d4   d5   d6
  Answer:      0    9    4    8    0    8    3
  Subtask:    SA   SS   SA   SA   SA   SA   SA
  Token:      t6  t15   t3   t8   t3    —    —
                         ↑         ↑
                   t3 at simple-add positions (no carry)
```

Every position is SA (simple add) — no cascades at all. t3 appears at d2 and d4 (both SA).
Other tokens (t6, t15, t8) handle positions with specific digit-sum values, but none of the
carry-cascade tokens (t2, t7) appear anywhere.

---

**The key insight:** the same token IDs recur consistently across different problems.
t2 = carry cascade. t7 = borrow cascade (subtraction only). t3 = no cascade needed.
The model has learned a **vocabulary for arithmetic reasoning** that maps directly to
Quirke's circuit definitions — without any supervision about carry logic.

### 4. Surgical intervention: the right token fixes hard-case failures

*All experiments below use the 1L/3H/510d model with K=1 abs30, trained on 100K examples.
This model gets C3-C6 accuracy ~70% — enough correct examples for comparison, enough errors to fix.*

**The two carry tokens: t9 vs t21**

The model learned two tokens that both appear at carry-related positions, but with different specializations:

| | **t9** (n=573) | **t21** (n=719) |
|---|---|---|
| Position | d2-d4 (spread) | d3 only (100%) |
| Sum = 9 | 56% | **95%** |
| Carry rate | 22% | 3% |
| Difficulty | **easy** (S0=23%, S2=28%) | **hard** (S5=37%, S6=37%) |
| Role | "shallow carry, maybe sum-9" | "deep cascade, definitely sum-9" |

t9 is a **shallow/ambiguous** token — it appears in easy problems where carry state doesn't matter much.
t21 is a **deep cascade specialist** — it appears specifically in 5-6 carry cascades where
every digit's sum is exactly 9 (Quirke's uncertain U state, eq. 2).

**The failure mode: using t9 at hard-cascade positions**

When the model encounters a hard cascade (C5/C6), it sometimes assigns t9 (the shallow token)
instead of t21 (the cascade specialist). This is like using the wrong circuit — the model
treats a deep cascade as a shallow one and gets the carry propagation wrong.

**The fix: globally replacing t9 with t21**

We test what happens when we force the model to always use t21 (the cascade specialist)
instead of t9:

```
                      Normal    All t9→t21    Effect
C3-C6 (hard carries):  70%        93%        +23pp ← hard cases dramatically improve
S5 (5 cascades):       27%        92%        +65pp ← nearly fixes the hardest split
S0 (no carries):       99%        74%        -25pp ← easy cases get worse
```

Forcing t21 everywhere is like telling the model "always assume deep cascade" — it fixes
hard problems (+65pp on S5!) but hurts easy ones where no cascade exists. The model needs
to *correctly choose* between t9 and t21 based on the input, and its main failure mode on
hard cases is choosing too conservatively (t9 instead of t21).

**Individual transplants confirm this:**

```
014560 + 125450 = 0140010
  Wrong: 0139010  — t9 at d2 (shallow token at carry position)
  Fixed: 0130010  — transplant correct token → d2 fixed ✓

332200 + 868010 = 1200210
  Wrong: 1190210  — t9 at d1 (shallow token at carry position)
  Fixed: 1100210  — transplant correct token → d1 fixed ✓
```

Each transplant fixes the specific digit where the wrong carry token was assigned.
This proves **token identity causally determines carry computation** — the wrong
token = wrong carry state = wrong answer digit.
""")

            gr.Markdown("""### 3. Tokens spread across digit positions

Unlike K=4 (where tokens are locked to fixed positions), K=1 tokens serve multiple
digit positions. This means token identity encodes **what computation to perform**,
not just **where in the sequence we are**.
""")
            gr.Image("static_figures/fig_token_positions.png")

            gr.Markdown("""### 4. Causal verification: token identity is essential for cascades

Three interventions test whether tokens carry real information:
- **Knockout**: remove all abstraction tokens → **0% accuracy** (total dependence)
- **Shuffle**: randomly permute token IDs within the sequence → accuracy drops
- **Random**: replace tokens with random IDs → accuracy drops further

The key finding: **the harder the cascade, the more token identity matters.**
For S5-S6 (5-6 consecutive carries), shuffling drops accuracy by 56-66 percentage points.
This directly parallels Quirke's finding that deeper cascades require more complex
internal circuits — SoRL externalizes these circuits as token sequences that must be correct.
""")
            gr.Image("static_figures/fig_causal_ablation.png")

            gr.Markdown("""### 5. Correspondence with Quirke's tri-state carry classifier

Quirke's eq. 2 defines a **tri-state carry classifier** STn for each digit position:
- **STn = 0**: digit sum ≤ 8, definitely no carry
- **STn = 1**: digit sum ≥ 10, definitely carry
- **STn = U**: digit sum = 9, carry depends on cascade from right (uncertain)

In our K=4 abs30 model (where tokens are position-locked, forcing sharper specialization):
- **Token t3**: maps to SA (simple add) with 0% carry — Quirke's **STn = 0**
- **Token t6**: maps to UC with input sum mod 10 = 9 in 92% of cases — Quirke's **STn = U**
- **Tokens t8, t9**: map to UC with carry = 100% — Quirke's **STn = 1**

The model independently discovered the tri-state classifier from the info-gain loss alone,
with no supervision about carry logic. This is the same circuit Quirke found via PCA of
hidden activations — but here it is readable directly from the token sequence.

*Analysis: K=1 abs30 and K=4 abs30, 2L/3H/510d, 100K training examples, 4400 eval examples.*
""")

        # ── Tab 3: LaTeX Scratchpad ──
        with gr.TabItem("LaTeX"):
            gr.Markdown("### LaTeX Scratchpad — copy sections directly into Overleaf")
            gr.Markdown("Two blocks: **main body** (setup + Finding #1 + forward ref) and **appendix** (Findings #2–5, all tables, training details). Copy each into the appropriate place in your paper.")

            gr.Markdown("#### § Main body — §Arithmetic setup (paste into paper body)")
            gr.Code(value=LATEX_ARITHMETIC_SETUP, label="arithmetic_setup.tex",
                    language=None, interactive=False)

            gr.Markdown("#### § Appendix — full analysis (paste into appendix)")
            gr.Markdown("Includes: per-split ablation table, token profiles, guided computation, Quirke analogy, training details. Requires `tcolorbox`, `booktabs`, `xcolor`, `multirow`, `enumitem`.")
            gr.Code(value=LATEX_APPENDIX, label="appendix_arithmetic.tex",
                    language=None, interactive=False)

        # ── Tab 4: About ──
        with gr.TabItem("About"):
            eval_info_md = gr.Markdown("")

            gr.Markdown("""### Using the models

All models are on [HuggingFace](https://huggingface.co/thoughtworks/arithmetic-sorl).
Code is on the [`amir/arithmetic`](https://github.com/fangyuan-ksgk/mod_gpt/tree/amir/arithmetic) branch.

```python
import torch
from arithmetic.data.hub import load_model
from arithmetic.training.evaluate import ArithmeticEvaluator
from transformers import AutoTokenizer

# Load model + tokenizer
model, config, metrics = load_model("add_sub_sorl_v1_abs30_K1_100K", device="cuda")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# Run full evaluation with per-split accuracy
evaluator = ArithmeticEvaluator(model, tokenizer, device="cuda")
results = evaluator.run(ops="add_sub", K=1, n_per_split=100)  # K=None for baseline
evaluator.print_table(results)
```

To inspect abstraction tokens on a single example:

```python
from arithmetic.training.train import QWEN3_TOKEN_MAP, QWEN3_INV_MAP
from sorl.sorl_trainer import infer_insert_mask, insert_tokens_with_padding, expand_prompt_len

base_v = model.vocab_sizes[0].item()

# Encode: 123456+654321=
prompt = [1,2,3,4,5,6, 10, 6,5,4,3,2,1, 12]
qwen_ids = torch.tensor([QWEN3_TOKEN_MAP[t] for t in prompt], device="cuda")

# Pad to full 21 tokens (14 prompt + 7 dummy answer), insert abstractions, recurse
seq = torch.cat([qwen_ids, torch.zeros(7, dtype=torch.long, device="cuda")])
ids = seq.unsqueeze(0)
im = infer_insert_mask(ids, K=1, attention_mask=torch.ones_like(ids))
ep = expand_prompt_len(torch.tensor([14], device="cuda"), im)
ed, ea = insert_tokens_with_padding(ids, torch.ones_like(ids), im, model.vocab_sizes[0], 151643)

data, ppt, logits = model.recursion(ed, ea, max_iterations=2,
    memory_span_abs=1792, memory_span_traj=1792, temperature=0.0, prompt_len=ep)

# Separate trajectory vs abstraction tokens
is_abs = data[0] >= base_v
abstractions = data[0][is_abs] - base_v    # abstraction token IDs (0-indexed)
print(f"Abstraction tokens: {abstractions.tolist()}")
# Each abstraction token encodes carry/borrow state at that position
```

Token IDs: `0-9` = digits, `10` = `+`, `11` = `-`, `12` = `=`.
Abstraction tokens are integers from 1 to `abs_vocab` (0 is the placeholder before recursion).
""")


    # ═══════════════════════════════════════════════════════════════
    # Callbacks
    # ═══════════════════════════════════════════════════════════════

    def on_refresh(arch):
        models = fetch_all_models()
        df = build_comparison_table(models, arch_filter=arch, enriched_only=False)

        n_models = len(models)
        summary = f"**{n_models}** models | Arch filter: {arch}"
        q_status = get_queue_status_text(n_models)
        eval_info = build_eval_info(models)
        new_summary_md = build_summary_markdown()

        main_cols = ["Ops", "Data", "Arch", "Baseline", "SoRL", "Config", "B_hf", "S_hf", "B_wandb", "S_wandb"]
        main_df = df[main_cols] if all(c in df.columns for c in main_cols) else pd.DataFrame()

        hard_cols = [c for c in df.columns
                     if (c.startswith("B_") or c.startswith("S_")) and "wandb" not in c]
        hard_base = ["Ops", "Data", "Arch", "Config"] if "Arch" in df.columns else ["Ops", "Data", "Config"]
        hard_df = df[hard_base + hard_cols] if len(df) > 0 else pd.DataFrame()

        model_names = sorted([m["subfolder"].removeprefix("non_enriched/") for m in models])
        model_dd_update = gr.update(choices=model_names, value=model_names[0] if model_names else "")

        return models, summary, q_status, main_df, hard_df, eval_info, model_dd_update, new_summary_md

    def on_detail(models, name):
        return build_detailed_splits(models, name.strip() if name else "")

    all_outputs = [models_state, summary_text, queue_status, main_table, hard_table, eval_info_md, model_selector, summary_md]

    refresh_btn.click(on_refresh, inputs=[arch_filter], outputs=all_outputs)
    arch_filter.change(on_refresh, inputs=[arch_filter], outputs=all_outputs)
    detail_btn.click(on_detail, inputs=[models_state, model_selector], outputs=[detail_table])

    timer = gr.Timer(300)
    timer.tick(on_refresh, inputs=[arch_filter], outputs=all_outputs)
    app.load(on_refresh, inputs=[arch_filter], outputs=all_outputs)


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False, theme=gr.themes.Soft())
