\documentclass{article}


% if you need to pass options to natbib, use, e.g.:
%     \PassOptionsToPackage{numbers, compress}{natbib}
\PassOptionsToPackage{numbers,compress}{natbib}

% before loading neurips_2024


% ready for submission
\usepackage{neurips_2024}



\usepackage[utf8]{inputenc} % allow utf-8 input
\usepackage[T1]{fontenc}    % use 8-bit T1 fonts
\usepackage{hyperref}       % hyperlinks
\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}
% \usepackage{natbib}         % for author-year citations
\usepackage{url}            % simple URL typesetting
\usepackage{booktabs}       % professional-quality tables
\usepackage{amsfonts}       % blackboard math symbols
\usepackage{nicefrac}       % compact symbols for 1/2, etc.
\usepackage{microtype}      % microtypography
\usepackage{xcolor}         % colors
% \usepackage{natbib}
% \usepackage{algorithmic}
\usepackage{amsmath}
\usepackage{amsthm}
\usepackage{algpseudocode}

\usepackage{booktabs}
\usepackage{multirow}
\usepackage{colortbl}
\usepackage{graphicx}
% \usepackage{afterpage}
\usepackage{float}

\usepackage{mathtools}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta,fit,backgrounds,decorations.pathreplacing,calligraphy,calc}

% ---- Analysis-section additions ----
\usepackage{wrapfig}
\graphicspath{{figures/}{../latex/figures/}}

% Family-tinted row colours for analysis tables.
\definecolor{qwenLight}{HTML}{E8F1FA}
\definecolor{qwenMid}{HTML}{D2E3F2}
\definecolor{qwenDark}{HTML}{B9D3E8}
\definecolor{llamaLight}{HTML}{FDE3C8}
\definecolor{llamaDark}{HTML}{F5C8A3}

% Editorial macros (providecommand to avoid clashes with packages).
\providecommand{\sorl}{\textsc{DLR}}
\providecommand{\sft}{\textsc{SFT}}
\providecommand{\svec}{\textsf{steering vector}}

\newtheorem{theorem}{Theorem}
\newtheorem*{theorem*}{Theorem}
% Custom restatement environment: prints "Theorem N (Title)." using \ref to appendix label.
\newenvironment{restatedthm}[2]%
  {\medskip\par\noindent\textbf{Theorem~\ref{#1}} (#2)\textbf{.}\itshape\quad}%
  {\par\upshape\medskip}
\newtheorem{lemma}{Lemma}
\newtheorem{corollary}{Corollary}
\newtheorem{definition}{Definition}
\newtheorem{remark}{Remark}
\newtheorem{assumption}{Assumption}

\usepackage[ruled,vlined]{algorithm2e}



\title{Dynamic Latent Routing}


% The \author macro works with any number of authors. There are two commands
% used to separate the names and addresses of multiple authors: \And and \AND.
%
% Using \And between authors leaves it to LaTeX to determine where to break the
% lines. Using \AND forces a line break at that point. So, if LaTeX puts 3 of 4
% authors names on the first line, and the last on the second line, try using
% \AND instead of \And before the third author name.


\author{%
  Fangyuan Yu \And Xin Su 
  \And Amir Abdullah\\
  % \texttt{fangyuan.yu18@gmail.com} \\
}

% Colored delta for ablation tables: red if negative, green if positive, black if zero.
\newcommand{\dlt}[1]{%
  \ifdim#1pt<0pt {\tiny\,\textcolor{red}{$(#1)$}}%
  \else\ifdim#1pt>0pt {\tiny\,\textcolor{green!55!black}{$(+#1)$}}%
  \else {\tiny\,$(\pm0.0)$}%
  \fi\fi}

\begin{document}


\maketitle

\begin{abstract}

We investigate the temporal concatenation of sub-policies in Markov Decision Processes with time-varying reward functions. We introduce the General Dijkstra Search (GDS) algorithm and prove that it discovers optimal goal-reaching policies by concatenating sub-policies optimal for intermediate goals, avoiding the per-state refinement that Bellman iteration requires.

Applying the ``search, select, update'' principle of GDS to language model post-training with discrete latent codes, we propose Dynamic Latent Routing (DLR), a method where a learned policy head searches over best code sequences during training whilst jointly updating the codes, the policy, and the model in a single stage, unlike prior methods that require multiple training phases. In the low-data fine-tuning regime, DLR matches or outperforms supervised fine-tuning across four datasets and five models, with a mean gain of $+8.0$ pp, while prior discrete-latent baselines lag behind SFT. Mechanistic analysis confirms that DLR learns task-specific routing patterns, and targeted ablations verify that individual codes carry distinct, causal effects on downstream performance.


\end{abstract}

\section{Introduction}

% ===== ORIGINAL INTRO (commented out) =====
% The sensory world we live in is continuous; the language we describe
% it with is discrete~\citep{harnad1990symbol}. Language models face the opposite setup: their
% input is discrete language, and their internal state is continuous.
% This raises a natural question: would a discrete code layer benefit
% a language model?
%
% This question is conceptually vague, as we don't even know under what formal
% setting one can even pose the question of whether discrete codes are necessary.
% A useful analogy is the sentence ``stand up and walk away''. It explicitly concatenates
% two policies along the temporal dimension. Moreover, ``stand up'' describes a policy
% that is optimal for a different goal than ``walk away''. What is being composed
% is therefore not just policies, but also goals. If we view a discrete code as a
% policy under a Markov Decision Process, then we must show the necessity of composing
% such policies.
%
% However, a standard Markov Decision Process $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$~\citep{sutton2018reinforcement}
% does not admit such structure. The globally optimal policy $\pi^{*}: \mathcal{S} \to \mathcal{A}$
% achieves maximal expected reward starting from any state. The goal for the policy
% is determined by the reward. When the reward doesn't vary with time, neither do the goals.
% The optimal policy here is obtained via Bellman Iteration~\citep{bellman1957dynamic}, where policies are updated
% on every state by iterating on all possible actions and picking the one believed to
% be most promising. This means the optimal policy doesn't explicitly depend on time,
% neither is the Bellman iteration process compositional.
%
% We found that by explicitly including a time-varying reward, globally optimal policies
% exist and they explicitly vary with time (Theorem 1). This motivates policy
% concatenation along the temporal domain. We further found that, if we define an optimal
% goal-reaching policy to be the policy that terminates in states within a goal set while
% achieving maximal reward, then a General Dijkstra Search (GDS) algorithm can provably find
% the optimal goal-reaching policy via composing policies optimal for other goals.
%
% This gives a plausible explanation of why discrete codes are desirable: composing
% them implements a ``learning algorithm'' that finds goal-oriented optimal policies.
% ===== END ORIGINAL =====

The sensory world we live in is continuous; the language we describe it with is discrete~\citep{harnad1990symbol}. Neural networks, increasingly capable across a wide range of tasks~\citep{radford2019gpt2,openai2025gpt55,anthropic2025claude47,deepseek2025v4,qwen2025qwen35,moonshot2025kimi26}, operate almost entirely in continuous representations even when their inputs are discrete. This raises a question: are discrete codes necessary?

Consider what a discrete code makes possible. The sentence ``stand up and walk away'' composes two policies along the temporal dimension: ``stand up'' and ``walk away'', each encoding a chain of low-level motor actions. Whilst hierarchical reinforcement learning~\citep{sutton1999options,dayan1993feudal,vezhnevets2017feudal,dietterich2000maxq,machado2023temporal} extends the agent's action space to include such sub-policies, a theoretical understanding of \emph{why} this helps is still lacking.

We fill this gap by making the reward in the MDP explicitly time-varying. In this setting, no static policy is optimal in general (Thm.~\ref{thm:static-gap}). We propose the General Dijkstra Search (GDS) algorithm, which searches over concatenations of sub-policies, and prove that it finds an optimal goal-reaching policy (Thm.~\ref{thm:gds}). GDS departs from the Bellman iteration paradigm that underpins most of RL: rather than refining a policy through per-state updates, it shows that searching over \emph{concatenations} of sub-policies has a guarantee of reaching the optimal goal-reaching policy.

Motivated by GDS, we study how an LLM can develop its own discrete latent codes. Given a natural language sequence $x$, an LLM maximizes its predictive likelihood $p_{\theta}(x)$. Injecting latent codes $a$ shifts the objective to maximizing the conditional likelihood $p_{\theta}(x \mid a)$. Unlike $x$, however, we have no supervision target for $a$; the codes become an action space the model must explore.

Prior work injecting discrete codes during post-training~\citep{goyal2023pause,pfau2024filler,su2025tokenassorted,zelikman2024metatokens,ramji2026abstractcot} shares two drawbacks. First, they inject codes as extra tokens into the natural language sequence, disrupting the structure seen during pre-training. This necessitates large training budgets to match baseline SFT and consistently lags behind in low-data regimes. We find that injecting codes directly as steering vectors into the residual stream avoids this disruption, matching or outperforming SFT under low-data constraints.

Second, these methods require multiple training stages: codes are fixed in advance~\citep{goyal2023pause,pfau2024filler,zelikman2024metatokens}, pre-labeled by a separate model~\citep{su2025tokenassorted}, or warmed up in a separate phase before search can begin~\citep{ramji2026abstractcot}. What GDS prescribes---jointly learning which codes to use, what they do, and how to compose them---is not achieved in a single stage by any existing method.

We address both issues with Dynamic Latent Routing (DLR). DLR can be viewed as a neural relaxation of GDS that preserves its core structure: \textbf{search} over candidate code sequences via guided rollouts from a policy head, \textbf{select} the sequence that maximizes the conditional likelihood $p_{\theta}(x \mid a)$, and \textbf{update} the policy head, codebook, and base model jointly from a single objective. The explicit priority queue is replaced by a learned policy head, but the ``search, select, update'' loop remains intact. By steering the residual stream rather than injecting tokens, DLR avoids disrupting the pre-trained sequence structure---matching or outperforming SFT in the low-data regime---while unifying code search and learning into a single stage. The learned codes are diverse, input-dependent, and carry causal effects on downstream behavior.

In summary, our main contributions are:
\begin{itemize}
    \item We prove that under a dynamic Markov Decision Process with time-varying rewards, optimal goal-reaching policies can be obtained by composing sub-policies via a novel General Dijkstra Search (GDS) algorithm.
    \item We introduce Dynamic Latent Routing (DLR), a single-stage post-training method where a learned policy head searches for discrete latent codes that steer the model's residual stream. DLR works well in the low-data post-training regime.
    \item DLR matches or outperforms SFT and discrete-latent baselines (Pause Tokens, TokenAssorted) across four QA benchmarks and five model sizes, with a mean gain of $+8.0$\,pp over SFT. Gains are largest on reasoning tasks: $+5.5$--$10.2$\,pp on GSM8K and $+7.3$--$12.9$\,pp on ScienceQA. The learned codes are diverse, structured, and causally load-bearing.
\end{itemize}

\section{Related Work}

% ===== ORIGINAL 2.1 / 2.2 (commented out) =====
% \subsection{Discrete latents for LLM computation}
% Recent work introduces discrete latents into LLMs via static abstraction tokens. Pause tokens \citep{goyal2023pause} prepend a fixed learned embedding to every input, so the abstraction is identical across contexts. TokenAssorted \citep{su2025tokenassorted} compresses text spans with a separately pre-trained VQ-VAE; the abstraction varies with context, but each span deterministically maps to one external code. Abstract-CoT \citep{ramji2026abstractcot} learns a reserved-vocabulary latent reasoning sequence via a multi-stage pipeline---masked-CoT SFT, self-distillation under constrained decoding, then RL. In all three, the abstraction is either static or learned through a separate warm-up stage; none performs a model-internal search jointly with the LM update. A separate line, representation engineering \citep{zou2023representation}, steers hidden states with hand-designed directions but freezes the backbone, so it is not directly comparable to fine-tuning methods like DLR or SFT. DLR lets the model search over its own latent codes via a per-step rollout in code-space, with the codebook trained end-to-end against the LM loss.
%
% This search inherits the mutual-information objective from unsupervised skill discovery (DIAYN~\citep{eysenbach2019diayn}, DADS~\citep{sharma2020dads}, RLP~\citep{hatamizadeh2025rlp}), where a skill-conditioned policy is trained against a \emph{fixed} skill prior. DLR departs from this assumption: the prior is replaced by the model's own routing head, so the abstractions and the policy that exploits them are learned and improved jointly throughout training.
%
% \subsection{Hierarchical and compositional reinforcement learning}
% Reinforcement learning methods that decompose a policy into composable sub-policies have developed along three lines. Hierarchical RL learns sub-policies through frameworks such as options \citep{sutton1999options, bacon2017option}, feudal networks \citep{dayan1993feudal, vezhnevets2017feudal}, and MAXQ \citep{dietterich2000maxq}, but composition relies on additional structure beyond the sub-policies themselves. Compositional and goal-conditioned RL \citep{schaul2015universal, andrychowicz2017hindsight, barreto2017successor, pmlr-v97-van-niekerk19a, cao2020zero, todorov2009compositionality, hunt2019composing} parameterizes the goal into the value function or defines an algebra over sub-policies, allowing a single agent to cover multiple tasks or transfer zero-shot to new goals, but assumes a stationary task reward. Non-stationary MDP analyses \citep{evendar2009online, lecarpentier2019non, cardoso2019algorithms} bound the regret incurred under time-varying rewards, but do not consider sub-policy composition. DLR operates in a dynamic MDP whose reward varies with the agent's internal state, and proves that a Dijkstra-style search over learned sub-policies returns goal-reaching and goal-covering optimal policies, joining the threads of sub-policy composition and time-varying reward under a single formal guarantee.
% ===== END ORIGINAL =====

\paragraph{Discrete latents in LLMs.} Recent work introduces discrete codes into LLMs as extra tokens. Filler Tokens~\citep{pfau2024filler} show that adding a single, repeated token can improve performance when training transformers on synthetic tasks. Pause Tokens~\citep{goyal2023pause} and Meta-Tokens~\citep{zelikman2024metatokens} extend this to post-training, but require extensive warm-up to match or exceed SFT. TokenAssorted~\citep{su2025tokenassorted} pre-labels fine-tuning sequences with diverse codes learned via a separate vector-quantized variational encoder before post-training the LLM on the interleaved sequence. Abstract-CoT~\citep{ramji2026abstractcot} uses a warm-up phase that iterates between two sub-phases to teach the model to use new tokens, then applies group relative preference optimization to refine code selection. Compared to DLR, none of these methods unifies code search, policy update, and LM update into a single training stage, and they all require substantial compute in a warm-up phase, limiting their application in the low-data fine-tuning regime; TokenAssorted and Abstract-CoT are additionally designed to compress chain-of-thought sequences, a direction not aligned with ours.

\paragraph{Representation engineering and steering.} Representation engineering~\citep{zou2023representation} manipulates hidden states to control model behavior, with applications in safety~\citep{yousefpour2025repbend,siu2025steeringsafety}, reasoning~\citep{tang2025glore,seal2025}, and truthfulness~\citep{whyrepeworks2025}. Existing methods either apply a single fixed steering vector; DLR learns dynamic steering codes jointly with the model through search.

\paragraph{Hierarchical and compositional reinforcement learning.} Bellman iteration~\citep{sutton2018reinforcement} has been the cornerstone of optimal policy search in reinforcement learning, refining a policy through per-state updates. Hierarchical RL introduces high-level actions through frameworks such as options~\citep{sutton1999options,bacon2017option}, feudal networks~\citep{dayan1993feudal,vezhnevets2017feudal}, and MAXQ~\citep{dietterich2000maxq}, but lacks a theoretical justification for why composing sub-policies along the temporal axis helps. Compositional and goal-conditioned RL~\citep{schaul2015universal,andrychowicz2017hindsight,barreto2017successor,pmlr-v97-van-niekerk19a,cao2020zero,todorov2009compositionality,hunt2019composing} parameterizes goals into value functions or defines algebraic operations over sub-policies, enabling multi-task learning and zero-shot transfer; however, their composition is not along the temporal dimension and assumes stationary rewards. In contrast, we explicitly define temporal policy concatenation in a dynamic MDP without introducing high-level actions, and prove that GDS discovers optimal goal-reaching policies by concatenating sub-policies optimal for intermediate goals. This provides the missing theoretical justification for composing sub-policies along the temporal axis.

\paragraph{Mechanistic interpretability and probing in LLMs.} Most existing LLM interpretability work recovers abstract structure from a model's internals after training. The mechanistic line identifies specific computational circuits \citep{olsson_2022_induction, wang_2023_ioi, nanda_2023_grokking, quirke_2024_addition, quirke_2024_addsub_preprint, zhang_2024_arithmetic}; probing reads semantic content out of hidden states \citep{li_2023_worldrepresentations, nanda_2023_linearworldmodels, belinkov_2022_probes, sun_2025_arithmeticerrors}; sparse autoencoder approaches train an extra decoder on top of the frozen model to extract 10k to 100k sparse features and then run a large automated pipeline to label them \citep{huben_2024_sae, paulo_2025_autointerp}. All of these methods assume the abstraction lies hidden inside the continuous representation, waiting to be recovered post hoc. DLR inverts this: the abstraction is baked in during training as $C$ abstraction codes produced by hard-argmax, each a directly observable, input-dependent routing decision; the same probing and auto-interpretation pipelines then apply to these codes at orders-of-magnitude lower cost.



% \section{Theory}
\section{Theoretical Foundation}
\label{sec:theoretical-foundation}

A Markov Decision Process is defined by the tuple $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$ with state space $\mathcal{S}$, action space $\mathcal{A}$, transition kernel $P(s; | s, a)$ and reward $r(s, a)$. Since the reward is not explicitly time-dependent, optimal policies are also time-invariant. To define policy concatenation along the temporal axis, we make the reward function explicitly time-dependent, yielding a dynamic MDP. We formalize policy concatenation, show how the value function decomposes under concatenation (Thm.~\ref{thm:concat-value}), and prove that optimal policies in this setting are in general time-dependent (Thm.~\ref{thm:static-gap}). Moreover, we show they can be obtained by dynamic Bellman iteration under mild assumptions (Thm.~\ref{thm:policy-iteration}), setting the stage for the General Dijkstra Search algorithm.


\begin{definition}[Dynamic Markov Decision Process (DMDP)]
A \emph{dynamic MDP} is a finite-horizon MDP $(\mathcal{S}, \mathcal{A}, P, \{r_t\}_{t=0}^{T-1}, \gamma)$ with state space $\mathcal{S}$, action space $\mathcal{A}$, time-homogeneous transition kernel $P(s'\mid s,a)$, time-varying reward $r_t:\mathcal{S}\times\mathcal{A}\to\mathbb{R}_{\le 0}$, discount $\gamma\in[0,1)$, and horizon $T\in\mathbb{N}$. WLOG we assume $r_t\le 0$: any $r_{\max}$-bounded reward shifts to $r_t-r_{\max}\le 0$ without changing the argmax, so reward can be read as \emph{negated cost}.
\end{definition}

\begin{definition}[Time-varying policy]
A \emph{time-varying} (or \emph{dynamic}) policy is a sequence 
$\pi = \{\pi_t\}_{t=0}^{T-1}$, where each
\[
\pi_t : \mathcal{S} \to \mathcal{A}
\]
is a (possibly deterministic) decision rule at time $t$.
Equivalently, we can write $\pi : \mathcal{S} \times \mathbb{N} \to \mathcal{A}$ and interpret
$\pi_t(\cdot) := \pi(\cdot, t)$.
We denote by
\[
\Pi := \bigl\{ \pi \,\big|\, \pi : \mathcal{S} \times \mathbb{N} \to \mathcal{A} \bigr\}
\]
the collection of all time-varying policies.
\end{definition}


\begin{definition}[Concatenated policy]\label{def:concat-policy}
Let $\pi^{1} \in \Pi_{T_{1}}$ and $\pi^{2} \in \Pi_{T_{2}}$ be time-varying
policies of horizons $T_{1}$ and $T_{2}$, respectively. Their concatenation
$\pi = \pi^{1}_{0:T_{1}} \circ \pi^{2}_{0:T_{2}} \in \Pi_{T_{1}+T_{2}}$ is the
policy defined by
\[
\pi_{t}(s)
=
\begin{cases}
\pi^{1}_{t}(s), & t = 0,\dots,T_{1}-1,\\
\pi^{2}_{t-T_{1}}(s), & t = T_{1},\dots,T_{1}+T_{2}-1.
\end{cases}
\]
\end{definition}

\begin{restatedthm}{thm:concat-value}{Value of a concatenated policy}
Under Assumptions~\hyperref[asm:A1]{A1}--\hyperref[asm:A3]{A3},
let $\pi = \pi^{1}_{0:T_{1}} \circ \pi^{2}_{0:T_{2}}$ be as in
Definition~\ref{def:concat-policy} and let $T := T_{1} + T_{2}$. Then for all
$t = 0,\dots,T-1$ and $s \in \mathcal{S}$,
\[
V_{t}^{\pi}(s)
=
V^{\pi^{1}}_{t}(s)
+ \gamma^{\max(T_{1}-t,0)}
  \,\mathbb{E}\!\Big[
      V^{\pi^{2}}_{\max(0,\,t-T_{1})}\bigl(s_{\max(t,T_{1})}\bigr)
      \,\Big|\, s_{t} = s
    \Big].
\]
The detailed proof (Appendix~\ref{thm:concat-value}) proceeds by splitting the discounted return at time $T_{1}$.
\end{restatedthm}

\subsection{Policy Composition and Goal-Oriented Optimality}
\label{sec:gds}

We formally define optimal goal reaching policies to be the policy with the largest value that reach a set of goal states. We then propose General Dijkstra Search, along with Thm.~\ref{thm:gds-reach} proving it discovers optimal goal reaching policies. 

\begin{definition}[Goal states]
For a policy $\pi \in \Pi_{t}$ starting from $s \in \mathcal{S}$, the goal
states $\mathcal{G}^{\pi}(s)$ are defined by
\[
\mathcal{G}^{\pi}(s) := \Big\{ g \in \mathcal{S} \,\big|\, p_{\pi}(s_{t}=g \mid s_{0}=s) > 0 \Big\}.
\]
\end{definition}

\begin{definition}[Reaching policies]
Given a start state $s \in \mathcal{S}$ and a set of goal states
$\mathcal{G} \subset \mathcal{S}$, the set of \emph{reaching policies}
$\Pi^{\mathcal{G}\supset}_{1:T}(s)$ is
\[
\Pi^{\mathcal{G}\supset}_{1:T}(s)
:= \Bigl\{ \pi \in \bigcup_{t=1}^{T} \Pi_{t} \,\Big|\, \mathcal{G}^{\pi}(s) \subset \mathcal{G} \Bigr\}.
\]
\end{definition}

\begin{definition}[Optimal goal-reaching policy]
Given a start state $s \in \mathcal{S}$ and a goal set
$\mathcal{G} \subset \mathcal{S}$, an \emph{optimal goal-reaching policy} is a
policy $\pi^{*} \in \Pi^{\mathcal{G}\supset}_{1:T}(s)$ such that, for all
$\pi \in \Pi^{\mathcal{G}\supset}_{1:T}(s)$,
\[
V_{0}^{\pi^{*}}(s) \geq V_{0}^{\pi}(s).
\]
\end{definition}

\begin{algorithm}[H]
\caption{General Dijkstra Search (Optimal Reach)}
\KwIn{Error tolerance $\epsilon_{t}:= \frac{r_{\max}}{1-\gamma} \cdot \sum_{i=t}^{\infty} \gamma^{i}$, start state $s \in \mathcal{S}$, goal states $\mathcal{G}^{*} \subset \mathcal{S}$}
Initialize $\mathcal{Q} = \{(\emptyset, 0, \{s\}, 0)\}$, $v = \emptyset$, $\mathcal{R} = \emptyset$\;
\While{$\mathcal{Q} \neq \emptyset$}{
Pop $(\pi_{1:t}, v_{t}, \mathcal{G}^{\pi_{1:t}}(s), t)$ from the priority queue with maximal value; add $\mathcal{G}^{\pi_{1:t}}(s)$ into $\mathcal{R}$ if it is not already inside\;
\If{$\mathcal{G}^{\pi_{1:t}}(s) \subset \mathcal{G}^{*}$}{
break\;
}
\If{$(\exists (s, \mathcal{G}) \in v \text{ s.t. } \mathcal{G} \subset \mathcal{G}^{\pi_{1:t}}(s) \text{ and } v_{t} \leq v(s,\mathcal{G}) - \epsilon_{t})$ or $t = T$}{
continue\;
}
\For{$\pi \in \Pi_{1}$}{
Concatenate policy $\pi_{1:t+1} = \pi_{1:t} \circ \pi$ and compute $v_{t+1} = v_{t} + \gamma^{t} \cdot \mathbb{E}_{s_{0:t}}\big[ V_{0}^{\pi}(s_{t}) \,\big|\, s_{0} = s\big]$\;
Push $(\pi_{1:t+1}, v_{t+1}, \mathcal{G}^{\pi_{1:t+1}}(s), t+1)$ into $\mathcal{Q}$\;
\ForEach{$(s, \mathcal{G}) \in v$ such that $\mathcal{G}^{\pi_{1:t+1}}(s) \subset \mathcal{G}$ and $\mathcal{G} \notin \mathcal{R}$}{
$v(s, \mathcal{G}) \gets \max(v(s, \mathcal{G}), v_{t+1})$\;
}
\If{$(s, \mathcal{G}^{\pi_{1:t}}) \notin v$}{
$v(s, \mathcal{G}^{\pi_{1:t}}) \gets v_{t+1}$\;
}
}
}
\end{algorithm}

\begin{restatedthm}{thm:gds-reach}{GDS optimality for optimal reach}
Under Assumptions~\hyperref[asm:A1]{A1}--\hyperref[asm:A3]{A3}, for any reachable goal set, General Dijkstra Search for Optimal Reach finds an optimal goal-reaching policy. Specifically, for every $\mathcal{G} \in \mathcal{G}_{T}^{\supset}(s)$, there exists $\pi^{*} \in \Pi_{1:T}^{\mathcal{G}\supset}(s)$ such that $V_{0}^{\pi^{*}}(s) \ge V_{0}^{\pi}(s)$ for every $\pi \in \Pi_{1:T}^{\mathcal{G}\supset}(s)$.
\end{restatedthm}

\noindent The proof (Appendix~\ref{thm:gds-reach}) follows from the queue invariant, a pruning guarantee, and the optimality of the first popped feasible policy.





\section{Methodology}
\label{sec:method}

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{Styles/figures/DLR.png}
  \caption{DLR method overview. \emph{Left:} chunk-level steering---each chunk $m$ is routed to a discrete code $a_m$, which indexes a steering vector $\alpha\, e_{a_m}$ from the codebook and is added in place to every hidden state $h_1,\dots,h_K$ in the chunk. \emph{Right:} per-step search---$N$ code sequences are sampled from the routing head (\textsc{search}), the one maximizing $p_\theta(x\mid a)$ is selected (\textsc{select}), and $\mathcal{L}_{\mathrm{DLR}}$ is used to jointly update the codes, routing head, and LM (\textsc{update}).}
  \label{fig:dlr-method-overview}
\end{figure}

Figure~\ref{fig:dlr-method-overview} gives the overall architecture. We
describe DLR in three pieces: the chunk-level discrete code that
indexes a steering vector applied inside the model
(\S\ref{sec:method:codes}); the three-component objective---a
generalist LM term, an information-gain reward, and a
mutual-information lower bound on $I(x;a)$---against which the model
and its routing head are trained (\S\ref{sec:method:loss}); and the
per-step search that picks which code sequence to optimize against
(\S\ref{sec:method:search}). Algorithm~\ref{alg:sorl-step} gives the
end-to-end training step.

\subsection{Discrete codes and chunk-level steering}
\label{sec:method:codes}

Let $\pi_{\theta}$ be a causal language model with hidden states
$h_t^{(l)}\in\mathbb{R}^{H}$ at block $l$ and token position $t$, and
let $l^{*}$ denote a chosen \emph{injection layer}. Given a
natural-language token sequence $x_{1:T}$, DLR maintains an
abstract-code sequence $a_{0:M-1}$ on a coarser temporal dimension:
each code steers a chunk of $K$ consecutive NL tokens at layer
$l^{*}$, where $K\in\mathbb{N}$ is the \emph{abstraction ratio}. The
chunk index of token~$t$ is
\begin{equation}\label{eq:chunkmap}
m(t) \;=\; \bigl\lfloor (t-1)/K \bigr\rfloor,
\qquad t\in\{1,\dots,T\},
\end{equation}
giving $M=\lceil T/K\rceil$ codes in total. Each chunk is first
\emph{routed} to a code and then \emph{steered} by the continuous
vector indexed by it (Fig.~\ref{fig:dlr-method-overview}, left).

\paragraph{Route: the routing head emits logits over codes.}
A linear head $W_{\mathrm{rt}}\in\mathbb{R}^{C\times H}$, with $C$
the codebook size, is applied at the \emph{first} token of each chunk
to produce routing logits:
\begin{equation}\label{eq:code}
z_m \;=\; W_{\mathrm{rt}}\,\mathrm{sg}\!\bigl(\,h_{mK+1}^{(l^{*})}\,\bigr)
\;\in\;\mathbb{R}^{C}.
\end{equation}
The role of the stop-gradient $\mathrm{sg}(\cdot)$ is discussed
alongside the loss in \S\ref{sec:method:loss}.

\paragraph{Steer: a vector indexed by the code is added to the hidden state.}
Each codebook entry $k\in\{0,\dots,C{-}1\}$ owns a learned vector
$e_k\in\mathbb{R}^{H}$, initialised to zero. At the same injection
layer, every token whose chunk index is $m$ has $\alpha\,e_{a_m}$
added in place:
\begin{equation}\label{eq:steer}
h_t^{(l^{*})}\;\leftarrow\;h_t^{(l^{*})} + \alpha\,e_{a_{m(t)}},
\qquad t\in\{1,\dots,T\},
\end{equation}
with $\alpha>0$ a fixed steering scale.

\subsection{Objective}
\label{sec:method:loss}

Write $x\equiv x_{1:T}$ for the NL sequence and $a\equiv a_{0:M-1}$
for the chunk-level code sequence. We use
$p_{\theta}(x)$ for the unsteered LM marginal,
$\hat{p}_{\theta}(x\mid a)$ for the LM conditional with codes
substituted via Eq.~\eqref{eq:steer}, and
$\hat{p}_{\theta}(a\mid x)$ for the routing head's distribution over
chunk codes read off from Eq.~\eqref{eq:code};
$\mathrm{sg}(\cdot)$ denotes stop-gradient.
The per-step objective is
\begin{equation}\label{eq:total-loss}
\begin{aligned}
\mathcal{L}_{\mathrm{DLR}}(x,a)
&\;=\;
\underbrace{-\log p_{\theta}(x)}_{\text{Generalist}}
\;-\;
\underbrace{\log\frac{\hat{p}_{\theta}(x\mid a)}{\mathrm{sg}\bigl(p_{\theta}(x)\bigr)}}_{\text{Information Gain}}
\;-\;\alpha_{\mathrm{policy}}\,
\underbrace{\log \hat{p}_{\theta}(a\mid x)}_{\text{Policy Optimization}} \\
&\;+\;\alpha_{\mathrm{reg}}\,
\underbrace{D_{\mathrm{KL}}\!\bigl(p_{\theta}(a)\,\big\|\,p_{\text{bi-zipf}}(a)\bigr)}_{\text{Marginal Entropy Regularization}}.
\end{aligned}
\end{equation}
\paragraph{Generalist.} $-\log p_{\theta}(x)$ is the standard LM loss, preserving the unsteered model's next-token behaviour. \paragraph{Information Gain.} The log-ratio
$\log\bigl[\hat{p}_{\theta}(x\mid a)/\mathrm{sg}(p_{\theta}(x))\bigr]$
is the loss-side information-gain term used in
RLP~\citep{hatamizadeh2025rlp}: it rewards code sequences that improve
the steered likelihood over the unsteered baseline, analogous to the
skill-conditioned intrinsic reward in DADS~\citep{sharma2020dads}.

\paragraph{Policy and Marginal Entropy.} The last two terms are the
DIAYN-style mutual-information loss
$-\log\hat{p}_{\theta}(a\mid x)+D_{\mathrm{KL}}(p_{\theta}(a)\|p(a))$
for unsupervised skill discovery~\citep{eysenbach2019diayn}, with
$p(a)$ instantiated as our bigram-Zipfian prior
$p_{\text{bi-zipf}}$ (Appendix~\ref{app:zipf}). The stop-gradient in
Eq.~\eqref{eq:code} keeps routing-side and LM-side gradients separated,
as in standard skill-discovery objectives~\citep{eysenbach2019diayn,sharma2020dads}.

\subsection{Per-step search over code sequences}
\label{sec:method:search}

The reward of a code sequence $a$ on an NL sequence $x$ is
$R_{\theta}(x,a)=\log p_{\theta}(x\mid a)$. The model has never been
trained to condition on $a$, so $p_{\theta}(x\mid a)$ is initially
flat in $a$: every code yields (near-)identical reward, and no amount
of policy improvement can tell them apart. Building the dependency of
$p_{\theta}(x\mid a)$ on $a$ and learning a policy that exploits it
are therefore the same problem---as in unsupervised skill
discovery~\citep{eysenbach2019diayn,sharma2020dads}---and DLR
addresses both with a per-step search (Algorithm~\ref{alg:sorl-step}),
a single-step relaxation of the General Dijkstra Search of
\S\ref{sec:gds} that returns the best-scoring rollout $a^{*}$.

\begin{algorithm}[t]
\caption{Dynamic Latent Routing (DLR).}
\label{alg:sorl-step}
Initialize $\theta$ (base LM, codebook $\{e_k\}$, routing head $W_{\mathrm{rt}}$)\;
Given a temperature schedule $t_{1:N}$ and total training steps $S$\;
\For{$s = 1$ \KwTo $S$}{
  Sample $a^{(i)} \sim \hat{p}_{\theta}^{\,t_i}(a\mid x)$ for $i\in\{1,\dots,N\}$\;
  Select $a^{*} = \arg\max_{a^{(i)}}\, p_{\theta}(x\mid a^{(i)})$\;
  Optimize $\mathcal{L}_{\mathrm{DLR}}(x,a^{*})$\;
}
\end{algorithm}






\section{Experiments}
\label{sec:experiments}

\paragraph{Setup.} We benchmark DLR against supervised fine-tuning
(SFT) on five base models---Qwen3-\{0.6B, 1.7B, 4B\} and
Llama3.2-\{1B, 3B\}---across four QA datasets: GSM8K
\citep{cobbe2021gsm8k} (math), ScienceQA (multi-domain science),
StrategyQA (multi-hop commonsense), and CSQA (commonsense). All runs use one epoch, $\mathrm{lr}=10^{-5}$, effective batch size
8. DLR is trained with $C{=}32$, abstraction
ratio $K{=}4$, $N{=}8$ rollouts and the loss of
\S\ref{sec:method:loss}. The $N$ rollouts are no-grad forward passes used
only to pick the winner $a^{*}$. All methods in Table~\ref{tab:main} are
run for the same number of optimiser steps on the same training split.

\paragraph{DLR outperforms every baseline at every (model, dataset)
cell.} Table~\ref{tab:main} reports test accuracy
(\#correct\,/\,split size) for DLR against three baselines:
supervised fine-tuning (SFT), pause tokens
\citep{goyal2023pause}, and TokenAssorted \citep{su2025tokenassorted}.
We use the bootstrap resampling method to construct 95\% confidence
intervals on the test sets; the results are shown in Table~\ref{tab:main}.
All methods are run in the same low-data regime: one epoch on the
matched training split. DLR is the top method in every
(model, dataset) cell, with a mean gain of $+8.0$\,pp over SFT. The
two competing latent-augmentation baselines do not uniformly improve
on SFT in this regime: pause tokens roughly match SFT on GSM8K/SciQA
and add a small margin on CSQA, while TokenAssorted catastrophically
degrades math/reasoning at smaller scales (e.g.\ $15.7$ vs.\ SFT
$46.0$ on GSM8K-Qwen3-0.6B; $17.3$ vs.\ SFT $56.4$ on
SciQA-Qwen3-1.7B). The TokenAssorted failure is informative rather
than incidental: replacing whole NL-token chunks with newly
introduced abstract tokens breaks the input sequence structure, and
\citet{su2025tokenassorted} recovers from this perturbation only by
training for many more epochs on far larger corpora; under
single-epoch low-data finetuning the abstract tokens never acquire
useful meaning. By contrast, DLR adds steering \emph{into} an
unaltered NL sequence and is therefore well-defined from step one.
DLR's gains are largest on the same math/reasoning benchmarks where
TokenAssorted collapses (GSM8K: $+5.5$\,pp on Qwen3-1.7B,
$+10.2$\,pp on Llama-1B, $+6.7$\,pp on Qwen3-4B), indicating that
data-dependent routing unlocks genuinely new capability without
disturbing the underlying language model.

\paragraph{Loss-term and search ablations.} App.~\ref{app:ablations}
reports the full ablation grid (5 models $\times$ 4 datasets) for each
component of the DLR objective in Eq.~\eqref{eq:total-loss} and for the
per-step search of Algorithm~\ref{alg:sorl-step}. Zeroing the
policy-optimization weight $\alpha_{\mathrm{policy}}$
(Table~\ref{tab:abl-alpha-abs}), replacing the (generalist $+$ information-gain)
pair with the plain specialist loss $-\log\hat{p}_{\theta}(x\mid a)$
(Table~\ref{tab:abl-alpha-base}), or zeroing the bigram-Zipfian
regulariser $\alpha_{\mathrm{reg}}$ (Table~\ref{tab:abl-alpha-zipf}) each costs
accuracy across most cells, with the policy term being the most load-bearing.
Reducing the rollout budget from $N{=}8$ (main table, Table~\ref{tab:abl-N8})
to $N{=}1$ (no search, Table~\ref{tab:abl-N1}) similarly degrades performance,
isolating the contribution of the per-step search beyond the loss alone.
Sweeping the sampling temperature
(Tables~\ref{tab:abl-temp0}--\ref{tab:abl-temp2}) shows the rollout policy
needs to be neither deterministic ($\tau{=}0$) nor uniform ($\tau{=}2$).

\begin{table}[t]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{5pt}
  \caption{Main results: test accuracy (\%) on four QA benchmarks.
    \textbf{DLR} is compared against SFT and two
    latent-augmentation baselines (pause tokens, TokenAssorted)
    matched in data and optimiser. \textbf{Bold}: best
    per (model, dataset). Subscripts are 95\%\,CI half-widths
    ($\pm$pp) from bootstrap resampling on the test sets.
    Split sizes are shown beneath each column header.}
  \label{tab:main}
  \begin{tabular}{l l cccc}
    \toprule
    Model & Method & GSM8K & SciQA & StrategyQA & CSQA \\
          &        & \scriptsize{(/1319)} & \scriptsize{(/2224)}
          & \scriptsize{(/687)} & \scriptsize{(/1221)} \\
    \midrule
    \multirow{5}{*}{Qwen3-0.6B}
      & SFT                  & $46.0_{\pm2.7}$ & $48.0_{\pm2.1}$ & $47.0_{\pm3.7}$ & $64.1_{\pm2.7}$ \\
      & PauseToken           & $46.2_{\pm2.7}$ & $47.9_{\pm2.1}$ & $24.6_{\pm3.2}$ & $64.7_{\pm2.7}$ \\
      & TokenAssorted        & $15.7_{\pm2.0}$ & $13.1_{\pm1.4}$ & $29.4_{\pm3.4}$ & $64.2_{\pm2.7}$ \\
      & DLR ($C{=}1$)       & $45.8_{\pm2.7}$ & $48.9_{\pm2.1}$ & $50.4_{\pm3.7}$ & $64.0_{\pm2.7}$ \\
      & \textbf{DLR}        & $\mathbf{49.4}_{\pm2.7}$ & $\mathbf{55.3}_{\pm2.1}$ & $\mathbf{54.6}_{\pm3.7}$ & $\mathbf{66.4}_{\pm2.6}$ \\
    \midrule
    \multirow{5}{*}{Qwen3-1.7B}
      & SFT                  & $60.2_{\pm2.6}$ & $56.4_{\pm2.1}$ & $51.7_{\pm3.7}$ & $74.3_{\pm2.5}$ \\
      & PauseToken           & $60.9_{\pm2.6}$ & $56.2_{\pm2.1}$ & $32.0_{\pm3.5}$ & $76.4_{\pm2.4}$ \\
      & TokenAssorted        & $28.5_{\pm2.4}$ & $17.3_{\pm1.6}$ & $33.0_{\pm3.5}$ & $76.3_{\pm2.4}$ \\
      & DLR ($C{=}1$)       & $63.5_{\pm2.6}$ & $60.0_{\pm2.0}$ & $55.3_{\pm3.7}$ & $77.4_{\pm2.3}$ \\
      & \textbf{DLR}        & $\mathbf{65.7}_{\pm2.6}$ & $\mathbf{64.1}_{\pm2.0}$ & $\mathbf{60.3}_{\pm3.7}$ & $\mathbf{78.4}_{\pm2.3}$ \\
    \midrule
    \multirow{5}{*}{Qwen3-4B}
      & SFT                  & $75.4_{\pm2.3}$ & $59.4_{\pm2.0}$ & $65.8_{\pm3.5}$ & $82.5_{\pm2.1}$ \\
      & PauseToken           & $78.6_{\pm2.2}$ & $63.8_{\pm2.0}$ & $37.6_{\pm3.6}$ & $80.2_{\pm2.2}$ \\
      & TokenAssorted        & $60.9_{\pm2.6}$ & $68.0_{\pm1.9}$ & $39.3_{\pm3.7}$ & $80.6_{\pm2.2}$ \\
      & DLR ($C{=}1$)       & $78.6_{\pm2.2}$ & $64.1_{\pm2.0}$ & $66.2_{\pm3.5}$ & $81.1_{\pm2.2}$ \\
      & \textbf{DLR}        & $\mathbf{82.1}_{\pm2.1}$ & $\mathbf{72.3}_{\pm1.9}$ & $\mathbf{68.7}_{\pm3.5}$ & $\mathbf{83.0}_{\pm2.1}$ \\
    \midrule
    \multirow{5}{*}{Llama3.2-1B}
      & SFT                  & $30.9_{\pm2.5}$ & $38.0_{\pm2.0}$ & $48.0_{\pm3.7}$ & $65.2_{\pm2.7}$ \\
      & PauseToken           & $31.3_{\pm2.5}$ & $34.0_{\pm2.0}$ & $25.5_{\pm3.3}$ & $64.9_{\pm2.7}$ \\
      & TokenAssorted        & $14.2_{\pm1.9}$ & $33.6_{\pm2.0}$ & $29.3_{\pm3.4}$ & $65.6_{\pm2.7}$ \\
      & DLR ($C{=}1$)       & $38.4_{\pm2.6}$ & $43.7_{\pm2.1}$ & $48.6_{\pm3.7}$ & $68.3_{\pm2.6}$ \\
      & \textbf{DLR}        & $\mathbf{41.1}_{\pm2.7}$ & $\mathbf{49.1}_{\pm2.1}$ & $\mathbf{51.4}_{\pm3.7}$ & $\mathbf{71.4}_{\pm2.5}$ \\
    \midrule
    \multirow{5}{*}{Llama3.2-3B}
      & SFT                  & $41.3_{\pm2.7}$ & $56.7_{\pm2.1}$ & $52.5_{\pm3.7}$ & $79.8_{\pm2.3}$ \\
      & PauseToken           & $41.5_{\pm2.7}$ & $57.6_{\pm2.1}$ & $29.0_{\pm3.4}$ & $77.1_{\pm2.4}$ \\
      & TokenAssorted        & $21.8_{\pm2.2}$ & $34.6_{\pm2.0}$ & $34.7_{\pm3.6}$ & $77.3_{\pm2.4}$ \\
      & DLR ($C{=}1$)       & $46.3_{\pm2.7}$ & $59.2_{\pm2.0}$ & $56.1_{\pm3.7}$ & $79.6_{\pm2.3}$ \\
      & \textbf{DLR}        & $\mathbf{49.1}_{\pm2.7}$ & $\mathbf{63.3}_{\pm2.0}$ & $\mathbf{62.3}_{\pm3.6}$ & $\mathbf{81.0}_{\pm2.2}$ \\
    \bottomrule
  \end{tabular}
\end{table}


% =================================================================
\section{Analysis}
\label{sec:analysis}

Having established downstream gains, we next ask what routing structure DLR
learns internally. We analyze the learned routes along three dimensions:
whether the codebook remains active, whether routing patterns depend on the
input, and whether the selected routes affect model behavior. These analyses
connect the accuracy gains in Table~\ref{tab:main} to the latent routing
mechanism learned by DLR.

% -------------------------------------------------------------
% \subsection{Codebook diversity}
\subsection{Does DLR learn an active codebook?}
\label{sec:diversity}

\begin{wraptable}[13]{r}{0.40\linewidth}
  \centering
  \footnotesize
  \begin{tabular}{lrr}
    \toprule
    Model        & cos   & util. \\
    \midrule
    \rowcolor{qwenLight}  Qwen3-0.6B   & $0.24$  & $41\%$ \\
    \rowcolor{qwenMid}    Qwen3-1.7B   & $0.28$  & $31\%$ \\
    \rowcolor{qwenDark}   Qwen3-4B     & $0.16$  & $56\%$ \\
    \rowcolor{llamaLight} Llama3.2-1B  & $0.01$  & $78\%$ \\
    \rowcolor{llamaDark}  Llama3.2-3B  & $0.00$  & $100\%$ \\
    \bottomrule
  \end{tabular}
  \caption{Codebook diversity (SciQA).}
  \label{tab:codebook-diversity}
\end{wraptable}

% We measure codebook diversity by the mean off-diagonal cosine similarity
% between learned steering vectors and by the fraction of codes used at least
% once on the SciQA test set. Lower cosine values indicate more orthogonal
% steering directions. The learned steering vectors are near-orthogonal (Llama
% $\lvert\bar{\cos}\rvert{\leq}0.01$, Qwen $0.16$--$0.28$), and codebooks are
% broadly utilized. Llama models use most slots, while Qwen codebooks are more
% concentrated.
A collapsed router would either reuse a single steering direction or leave most
codes unused. We therefore measure codebook activity in two ways: the mean
off-diagonal cosine similarity between learned steering vectors, and the
fraction of codes selected at least once on the SciQA test set.
% Table~\ref{tab:codebook-diversity} shows that the learned steering vectors
% are close to orthogonal, while a substantial fraction of the codebook is used
% across models.
Table~\ref{tab:codebook-diversity} shows that the learned steering vectors are
close to orthogonal (mean off-diagonal cosine: Llama $\leq0.01$, Qwen
$0.16$--$0.28$), while code usage remains non-collapsed across models
($31$--$100\%$ utilization). This indicates that DLR learns an active set of
latent routes rather than relying on a single shared steering direction. We
next examine how this codebook is used across inputs by measuring topic
specialization in code n-grams.

% -------------------------------------------------------------
% \subsection{N-gram topic specialization}
\subsection{Are routing patterns input-dependent?}
\label{sec:ngram}

% \emph{Topic purity} of a code \textsf{n-gram} is its share of occurrences in the single most frequent topic (null $\approx\!0.17$). Fig.~\ref{fig:ngram-purity} shows the fraction of \textsf{n-gram}s above purity $\tau$ vs.\ length $N$: all curves sit well above null and rise with $N$. Qwen3-4B holds soft ($\tau{\geq}0.30$) but loses hard ($\tau{\geq}0.90$) specialization, consistent with its broader utilization (\S\ref{sec:diversity}).
To test whether code usage varies with the input, we measure topic purity for
code n-grams. For each code n-gram, topic purity is the fraction of its
occurrences assigned to the single most frequent SciQA topic; a topic-independent
routing pattern would stay close to the topic prior (approximately $0.17$).
Figure~\ref{fig:ngram-purity} shows that code n-grams are consistently more
topic-pure than this baseline, and that purity increases with n-gram length.
This indicates that DLR uses its codebook in structured, input-dependent
sequences rather than assigning codes independently of content.

\begin{figure}[!t]
  \centering
  \includegraphics[width=0.82\linewidth]{ngram_purity_sweep_all}
  \caption{Fraction of \textsf{n-gram}s (occurring ${\geq}30\times$) with topic purity ${\geq}\tau$, vs.\ length $N$. Panels: $\tau\in\{0.30,0.50,0.75,0.90\}$; null $\approx 0.17$.}
  \label{fig:ngram-purity}
\end{figure}

% -------------------------------------------------------------
% \subsection{Necessity: global perturbation}
\subsection{Do the selected routes affect model behavior?}
\label{sec:necessity}

% Zeroing the learned steering vectors (\emph{scale\,$\to$\,0}) or scrambling the router (\emph{random-replace}) drops SciQA accuracy by $3.4$--$12.9$\,pp and $2.6$--$11.3$\,pp respectively (Table~\ref{tab:global-ablation}). Routing is load-bearing, not ornamental.
Having shown that DLR learns active and input-dependent routes, we test whether
those routes affect model predictions.

\paragraph{Global perturbation.}
We first perturb the routing mechanism at the model level. Setting the steering
scale to zero or replacing routed codes with random alternatives reduces SciQA
accuracy across all models: the scale-zero intervention costs $3.4$--$12.9$\,pp,
while random replacement costs $2.6$--$11.3$\,pp
(Table~\ref{tab:global-ablation}). This shows that the routing mechanism
contributes directly to downstream performance.

% \begin{table}[h]
%   \centering
%   \begin{tabular}{lccc}
%     \toprule
%     Model & steered & scale $\to 0$ & random-replace \\
%     \midrule
%     Qwen3-0.6B & $55.3\%$ & $49.1\%$ & $50.5\%$ \\
%     Qwen3-1.7B & $64.1\%$ & $57.7\%$ & $58.4\%$ \\
%     Qwen3-4B    & $72.3\%$ & $60.1\%$ & $64.2\%$ \\
%     Llama3.2-1B & $49.1\%$ & $36.2\%$ & $37.8\%$ \\
%     Llama3.2-3B & $63.3\%$ & $59.9\%$ & $60.7\%$ \\
%     \bottomrule
%   \end{tabular}
%   \caption{SciQA test accuracy under global ablation modes.}
%   \label{tab:global-ablation}
% \end{table}

% -------------------------------------------------------------
% \subsection{Controllability: targeted code ablation}
% \label{sec:controllability}

\paragraph{Single-code ablation.}
% Ablating codes \emph{one at a time} (random-swap, full SciQA) is uniformly harmful in aggregate (Table~\ref{tab:percode-acc-summary}) --- no code is dispensable.
We then ablate codes one at a time. Single-code interventions are harmful in
aggregate, with mean drops of $0.71$--$2.74$\,pp across the tested Qwen models
(Table~\ref{tab:percode-acc-summary}). This indicates that individual codes are
also load-bearing, not only the routing mechanism as a whole.

\begin{table}[H]
  \centering
  \footnotesize
  \begin{minipage}[t]{0.53\linewidth}
    \centering
    \setlength{\tabcolsep}{3.6pt}
    \begin{tabular}{lccc}
      \toprule
      Model & steered & scale $\to 0$ & random-replace \\
      \midrule
      Qwen3-0.6B & $55.3\%$ & $49.1\%$ & $50.5\%$ \\
      Qwen3-1.7B & $64.1\%$ & $57.7\%$ & $58.4\%$ \\
      Qwen3-4B & $72.3\%$ & $60.1\%$ & $64.2\%$ \\
      Llama3.2-1B & $49.1\%$ & $36.2\%$ & $37.8\%$ \\
      Llama3.2-3B & $63.3\%$ & $59.9\%$ & $60.7\%$ \\
      \bottomrule
    \end{tabular}
    \caption{SciQA test accuracy under global ablation modes.}
    \label{tab:global-ablation}
  \end{minipage}\hfill
  \begin{minipage}[t]{0.42\linewidth}
    \centering
    \setlength{\tabcolsep}{5.0pt}
    \begin{tabular}{lccc}
      \toprule
      Model & mean $\Delta$ & min & max \\
      \midrule
      \rowcolor{qwenLight} Qwen3-0.6B & $-2.74$ & $-3.90$ & $-1.43$ \\
      \rowcolor{qwenMid}   Qwen3-1.7B & $-0.71$ & $-1.19$ & $-0.19$ \\
      \rowcolor{qwenDark}  Qwen3-4B   & $-1.06$ & $-1.48$ & $-0.57$ \\
      \bottomrule
    \end{tabular}
    \caption{Single-code ablation: SciQA $\Delta$acc (pp) over four codes per model.}
    \label{tab:percode-acc-summary}
  \end{minipage}
\end{table}

% \begin{wraptable}{r}{0.46\linewidth}
%   \centering
%   \vspace{-\baselineskip}
%   \caption{Single-code ablation: SciQA $\Delta$acc (pp) over four codes per model.}
%   \label{tab:percode-acc-summary}
%   \small
%   \begin{tabular}{lccc}
%     \toprule
%     Model & mean $\Delta$ & min & max \\
%     \midrule
%     \rowcolor{qwenLight} Qwen3-0.6B & $-2.74$ & $-3.90$ & $-1.43$ \\
%     \rowcolor{qwenMid}   Qwen3-1.7B & $-0.71$ & $-1.19$ & $-0.19$ \\
%     \rowcolor{qwenDark}  Qwen3-4B   & $-1.06$ & $-1.48$ & $-0.57$ \\
%     \bottomrule
%   \end{tabular}
% \end{wraptable}

% Per-topic, the aggregate masks sign-flipping effects (Fig.~\ref{fig:ablation-topic-delta}): ablating code~0 lifts \emph{biology} $+3.6$\,pp on Qwen3-0.6B (while dropping \emph{physics} $-4.0$\,pp), \emph{writing-strategies} $+4.8$\,pp on Qwen3-1.7B, and \emph{chemistry} $+9.8$\,pp on Qwen3-4B. Each code is a \emph{topic-conditional} handle.
\paragraph{Topic-level effects.}
Finally, we examine how single-code ablations vary by SciQA topic. Ablating
code~0 increases \emph{biology} accuracy by $+3.6$\,pp on Qwen3-0.6B while
reducing \emph{physics} accuracy by $-4.0$\,pp; on larger Qwen models, the same
intervention improves \emph{writing-strategies} by $+4.8$\,pp on Qwen3-1.7B and
\emph{chemistry} by $+9.8$\,pp on Qwen3-4B (Fig.~\ref{fig:ablation-topic-delta}).
Thus individual codes act as topic-conditional handles rather than uniform
performance knobs.

\begin{figure}[H]
  \centering
  \includegraphics[width=0.95\linewidth]{fig_ablation_topic_delta}
  \caption{Per-topic $\Delta$acc (pp) under single-code ablation. \textbf{Code~0} in red, other codes as gray backdrop. Spokes outside the zero ring: ablation helps; inside: hurts.}
  \label{fig:ablation-topic-delta}
\end{figure}

\section{Conclusion}

We present DLR as a single-stage method for learning discrete latent routes
inside language models. The approach is grounded in a dynamic-MDP view of
policy composition and trains chunk-level routing decisions, steering vectors,
and the base model jointly. In low-data QA fine-tuning, DLR improves over SFT
and prior discrete-latent baselines. Our analysis shows that the learned
codebook is active, input-dependent, and causally involved in model
predictions. These results suggest that discrete latent routing can serve as a
practical mechanism for learning compositional internal control in language
models.

\bibliographystyle{plainnat}
\bibliography{reference}

\appendix

\section{Bigram-Zipfian prior}
\label{app:zipf}

The Marginal Entropy Regularization term in the DLR objective
(Eq.~\eqref{eq:total-loss}) is the KL divergence from the mini-batch's
empirical bigram distribution over consecutive chunk codes
$(a_m, a_{m+1})$ to a fixed bigram-Zipfian prior $p_{\text{bi-zipf}}$
on $C^{2}$:
\begin{equation}\label{eq:zipf}
D_{\mathrm{KL}}\!\bigl(p_{\theta}(a)\,\big\|\,p_{\text{bi-zipf}}(a)\bigr)
\;=\;
  \mathrm{KL}\!\left(\,p(a_m,a_{m+1})\;\Big\|\;p_{\text{bi-zipf}}\,\right).
\end{equation}
The prior puts a Zipfian rank-frequency profile on single codes and
penalises consecutive-code repetition; this maintains a heavy-tailed
but compositional usage pattern, with a small core of codes appearing
frequently while the tail remains active.

\section{Theory for Dynamic MDP and Existence of Optimal Policy}
\begin{definition}[Dynamic value function and Q-function]
Given a dynamic policy $\pi \in \Pi$, its (time-indexed) value function
$V^{\pi} : \mathcal{S} \times \{0,\dots,T\} \to \mathbb{R}$ is defined by
\[
V^{\pi}_{t}(s)
:= \mathbb{E}_{\pi}\Bigl[ \sum_{i=t}^{T} \gamma^{\,i-t} \,
    r_{i}(s_{i}, a_{i}) \,\Big|\, s_{t} = s \Bigr],
\]
where $a_i = \pi_i(s_i)$ for $i \ge t$, and the expectation is taken over
trajectories $(s_t,a_t,\dots,s_T,a_T)$ induced by $\pi$ and $P$.

Its (time-indexed) Q-function $Q^{\pi} : \mathcal{S} \times \mathcal{A}
\times \{0,\dots,T\} \to \mathbb{R}$ is defined by
\[
Q_{t}^{\pi}(s,a)
:= \mathbb{E}_{\pi}\Bigl[ \sum_{i=t}^{T} \gamma^{\,i-t} \,
    r_{i}(s_{i}, a_{i}) \,\Big|\, s_{t} = s,\, a_{t} = a \Bigr],
\]
where $a_t$ is fixed to be $a$ and $a_i = \pi_i(s_i)$ for $i > t$.
\end{definition}

\begin{definition}[Finite-horizon truncation]
Let $\pi = (\pi_{0},\pi_{1},\dots,\pi_{T-1})$ be a time-varying policy with
horizon $T$. For any $n \le T$, define the truncated policy by
\[
\pi_{0:n} := (\pi_{0},\pi_{1},\dots,\pi_{n-1}).
\]
\end{definition}

\begin{definition}[Finite-horizon value function and Q-function]
Let $\pi = (\pi_{0},\pi_{1},\dots,\pi_{T-1})$ be a time-varying policy. Its
finite-horizon value function and Q-function are defined, for $t=0,\dots,T-1$,
by
\[
V_{t}^{\pi}(s)
:= \mathbb{E}\Bigg[
    \sum_{i=t}^{T-1} \gamma^{\,i-t}\, r_{i}(s_{i}, a_{i})
    \,\Big|\, s_{t}=s,\ a_{i}=\pi_{i}(s_{i})
\Bigg],
\]
\[
Q_{t}^{\pi}(s,a)
:= r_{t}(s,a)
 + \mathbb{E}\Bigg[
    \sum_{i=t+1}^{T-1} \gamma^{\,i-t}\, r_{i}(s_{i}, a_{i})
    \,\Big|\, s_{t}=s,\ a_{t}=a,\ a_{i}=\pi_{i}(s_{i})
\Bigg].
\]
For a truncated policy $\pi_{0:n}$, the same formulas apply with terminal time
$n-1$ in place of $T-1$.
\end{definition}

\begin{theorem}[Bellman equations for finite-horizon time-varying policies]
For any finite-horizon time-varying policy $\pi$,
\[
V^{\pi}_{t}(s)
= r_{t}\bigl(s,\pi_{t}(s)\bigr)
  + \gamma \sum_{s'} p\bigl(s' \mid s,\pi_{t}(s)\bigr) V^{\pi}_{t+1}(s'),
\quad t=0,\dots,T-1,
\]
and
\[
Q_{t}^{\pi}(s,a)
= r_{t}(s,a)
  + \gamma \sum_{s'} p(s' \mid s,a) V^{\pi}_{t+1}(s'),
\quad t=0,\dots,T-1.
\]
The same identities hold for truncated policies on their corresponding finite
horizon.
\end{theorem}

\begin{theorem}[Dynamic Bellman equations]

Given any (possibly time-dependent) policy $\pi = \{\pi_t\}_{t \ge 0}$ within a dynamic MDP environment $(\mathcal{S}, \mathcal{A}, P, \{r_t\}_{t \ge 0}, \gamma)$, its value function $V^{\pi}$ and Q-function $Q^{\pi}$ satisfy the (dynamic)
Bellman equations:
\begin{align}
Q_{t}^{\pi}(s,a)
&= r_{t}(s,a) + \gamma \sum_{s'} P(s' \mid s,a)\, V_{t+1}^{\pi}(s'), \\
V^{\pi}_{t}(s)
&= \sum_{a} \pi_{t}(a \mid s)\, Q^{\pi}_{t}(s,a), \\
V^{\pi}_{t}(s)
&= \sum_{a} \pi_{t}(a \mid s)\Bigl[
    r_{t}(s,a)
    + \gamma \sum_{s'} P(s' \mid s,a)\, V^{\pi}_{t+1}(s')
\Bigr].
\end{align}
\end{theorem}

\begin{proof}
We start from the definition of $Q_t^{\pi}(s,a)$:
\begin{align*}
Q_t^{\pi}(s,a)
&= \mathbb{E}_{\pi}\Bigl[\sum_{i=t}^{\infty} \gamma^{i-t} \, r_i(s_i, a_i)
\,\Big|\, s_t = s, a_t = a \Bigr] \\
&= \mathbb{E}_{\pi}\Bigl[
r_t(s_t,a_t)
+ \sum_{i=t+1}^{\infty} \gamma^{i-t} \, r_i(s_i, a_i)
\,\Big|\, s_t = s, a_t = a \Bigr] \\
&= r_t(s,a)
+ \underbrace{\mathbb{E}_{\pi}\Bigl[
\sum_{i=t+1}^{\infty} \gamma^{i-t} \, r_i(s_i, a_i)
\,\Big|\, s_t = s, a_t = a \Bigr]}_{(I)}.
\end{align*}

We now simplify term $(I)$. First factor out $\gamma$ and re-index the powers:
\begin{align*}
(I)
&= \mathbb{E}_{\pi}\Bigl[
\sum_{i=t+1}^{\infty} \gamma^{i-t} \, r_i(s_i, a_i)
\,\Big|\, s_t = s, a_t = a \Bigr] \\
&= \mathbb{E}_{\pi}\Bigl[
\sum_{i=t+1}^{\infty} \gamma \, \gamma^{i-(t+1)} \, r_i(s_i, a_i)
\,\Big|\, s_t = s, a_t = a \Bigr] \\
&= \gamma \, \mathbb{E}_{\pi}\Bigl[
\sum_{i=t+1}^{\infty} \gamma^{i-(t+1)} \, r_i(s_i, a_i)
\,\Big|\, s_t = s, a_t = a \Bigr].
\end{align*}

Apply the tower property of conditional expectation, conditioning on $s_{t+1}$:
\begin{align*}
(I)
&= \gamma \, \mathbb{E}_{\pi}\Bigl[
\mathbb{E}_{\pi}\Bigl[
\sum_{i=t+1}^{\infty} \gamma^{i-(t+1)} \, r_i(s_i, a_i)
\,\Big|\, s_{t+1} \Bigr]
\,\Big|\, s_t = s, a_t = a \Bigr].
\end{align*}

By the definition of the time-varying value function $V_{t+1}^{\pi}$, we have
\[
V_{t+1}^{\pi}(s')
= \mathbb{E}_{\pi}\Bigl[
\sum_{i=t+1}^{\infty} \gamma^{i-(t+1)} \, r_i(s_i, a_i)
\,\Big|\, s_{t+1} = s' \Bigr].
\]
Thus,
\begin{align*}
(I)
&= \gamma \, \mathbb{E}_{\pi}\Bigl[
V_{t+1}^{\pi}(s_{t+1})
\,\Big|\, s_t = s, a_t = a \Bigr] \\
&= \gamma \sum_{s'} P(s' \mid s,a) \, V_{t+1}^{\pi}(s').
\end{align*}

Substituting this back into the expression for $Q_t^{\pi}(s,a)$ yields
\[
Q_t^{\pi}(s,a)
= r_t(s,a) + \gamma \sum_{s'} P(s' \mid s,a) \, V_{t+1}^{\pi}(s'),
\]
which proves the first Bellman equation.

Next, consider the value function $V_t^{\pi}(s)$:
\begin{align*}
V_t^{\pi}(s)
&= \mathbb{E}_{\pi}\Bigl[
\sum_{i=t}^{\infty} \gamma^{i-t} \, r_i(s_i, a_i)
\,\Big|\, s_t = s \Bigr].
\end{align*}
Condition on the first action $a_t$ chosen according to $\pi_t(\cdot \mid s)$ and
use the tower property:
\begin{align*}
V_t^{\pi}(s)
&= \mathbb{E}_{\pi}\Bigl[
\mathbb{E}_{\pi}\Bigl[
\sum_{i=t}^{\infty} \gamma^{i-t} \, r_i(s_i, a_i)
\,\Big|\, s_t = s, a_t \Bigr]
\,\Big|\, s_t = s \Bigr] \\
&= \sum_{a} \pi_t(a \mid s) \,
\mathbb{E}_{\pi}\Bigl[
\sum_{i=t}^{\infty} \gamma^{i-t} \, r_i(s_i, a_i)
\,\Big|\, s_t = s, a_t = a \Bigr] \\
&= \sum_{a} \pi_t(a \mid s) \, Q_t^{\pi}(s,a),
\end{align*}
which proves the second equation.

Finally, substituting the first equation into the second, we obtain
\begin{align*}
V_t^{\pi}(s)
&= \sum_{a} \pi_t(a \mid s) \, Q_t^{\pi}(s,a) \\
&= \sum_{a} \pi_t(a \mid s)
\Bigl[ r_t(s,a) + \gamma \sum_{s'} P(s' \mid s,a) \, V_{t+1}^{\pi}(s') \Bigr],
\end{align*}
which is the third equation. This completes the proof.
\end{proof}

\begin{definition}[Time-varying Bellman policy operator]
The \emph{time-varying Bellman policy operator}
\[
\mathit{TP} : \Pi \to \Pi
\]
is defined for any policy $\pi \in \Pi$, time index $t \in \mathbb{N}$, and state
$s \in \mathcal{S}$ by
\begin{align*}
(\mathit{TP} \cdot \pi)_{t}(s)
&:= \arg\max_{a \in \mathcal{A}} Q_{t}^{\pi}(s,a) \\
&= \arg\max_{a \in \mathcal{A}}
\Bigl[
    r_{t}(s,a)
    + \gamma \sum_{s'} p(s' \mid s,a)\, V_{t+1}^{\pi}(s')
\Bigr].
\end{align*}
\end{definition}
Intuitively, Bellman policy operator asks a policy to do what it believed to be its best choice at any point in time. 

\begin{lemma}
Let $\pi = \{\pi_{t}\}_{t=0}^{T}$ be a (time-varying) policy.
For any integers $n \in \mathbb{N}$ and $m \ge 1$, and any state $s \in \mathcal{S}$,
the value $V_{n}^{\pi}(s)$ can be written as a linear combination of 
$\{V_{n+m}^{\pi}(s')\}_{s' \in \mathcal{S}}$:
\[
V_{n}^{\pi}(s)
= \sum_{s' \in \mathcal{S}} C\bigl(\pi_{n:n+m-1}, s'\bigr)\, V_{n+m}^{\pi}(s')
  + C\bigl(\pi_{n:n+m-1}\bigr),
\]
where the coefficients $C(\pi_{n:n+m-1}, s') > 0$ and $C(\pi_{n:n+m-1})$ depend
only on the policy slice $\{\pi^{t}\}_{t=n}^{n+m-1}$ and the dynamics, but not on 
$V_{n+m}^{\pi}$.
\end{lemma}

\begin{proof}
Denote by $\pi^{n}$ the decision rule at time $n$, i.e.\ $a = \pi^{n}(s)$ for $s \in \mathcal{S}$.
By the (dynamic) Bellman equation, we have
\begin{align*}
V_{n}^{\pi}(s)
&= Q_{n}^{\pi}\bigl(s, \pi^{n}(s)\bigr) \\
&= r_{n}\bigl(s, \pi^{n}(s)\bigr)
   + \gamma(n+1) \sum_{s'} p\bigl(s' \mid s, \pi^{n}(s)\bigr)\,
     V_{n+1}^{\pi}(s').
\end{align*}
This shows that $V_{n}^{\pi}(s)$ is a linear combination of the values
$\{V_{n+1}^{\pi}(s')\}_{s' \in \mathcal{S}}$, where the coefficients (including the
constant term) depend only on $\pi^{n}$ and the transition kernel $p$.

By repeatedly unrolling this relation from time $n$ up to time $n+m-1$, we obtain
an expression of the form
\[
V_{n}^{\pi}(s)
= \sum_{s' \in \mathcal{S}} C\bigl(\pi_{n:n+m-1}, s'\bigr)\, V_{n+m}^{\pi}(s')
  + C\bigl(\pi_{n:n+m-1}\bigr),
\]
where the coefficients $C(\pi_{n:n+m-1}, s')$ and $C(\pi_{n:n+m-1})$ are determined
by the sequence of policies $\{\pi^{t}\}_{t=n}^{n+m-1}$, the transition probabilities
and the rewards. Since the transition probabilities and discount factors are nonnegative,
each $C(\pi_{n:n+m-1}, s')$ is nonnegative; in particular, for states that are reachable
under the policy slice $\pi_{n:n+m-1}$, these coefficients are strictly positive.

This completes the proof.
\end{proof}


\begin{theorem}[Policy Improvement Theorem]
Given a deterministic policy $\pi$ in a dynamic MDP, the Bellman policy
operator $\mathit{TP}$ improves the value function in the sense that
\[
V^{\mathit{TP} \cdot \pi}_{t}(s) \;\ge\; V^{\pi}_{t}(s)
\quad \forall\, s \in \mathcal{S},\ \forall\, t \in \mathbb{N}.
\]
\end{theorem}

\begin{proof}

Let $\pi^{\infty} := \mathit{TP} \cdot \pi$ denote the greedy policy obtained
by the time-varying Bellman policy operator, and write its $t$-th decision rule
as $\pi_t^{\infty}$.

For each $n \in \mathbb{N}$, define a \emph{hybrid} deterministic policy
$\pi^{(n)}$ which follows $\pi^{\infty}$ up to time $n$, then reverts to $\pi$:
\[
\pi^{(n)} := \bigl[\,\pi_1^{\infty}, \dots, \pi_n^{\infty},\,
\pi_{n+1}, \pi_{n+2}, \dots \bigr].
\]
We also set $\pi^{(0)} := \pi$. Note that $\pi^{(n)}$ and $\pi^{(n-1)}$
differ only at time $n$.

\textit{Step 1: Improvement at time $n$}

Consider the value functions at time $n$. Since $\pi^{(n-1)}_{n:T} = \pi_{n:T}$, we know 
$$
V_{n}^{\pi^{(n-1)}}(s) = Q_{n}^{\pi}\bigl(s, \pi_n(s)\bigr)
$$
as unrolling $\pi^{(n-1)}$ from time $n$ onward is the same as unrolling $\pi$ from time $n$ onward. For the hybrid policy $\pi^{(n)}$, applying dynamic Bellman equation produces
\begin{align*}
V_{n}^{\pi^{(n)}}(s)
&= Q_{n}^{\pi^{(n)}}\bigl(s, \pi^{(n)}_n(s)\bigr) \\
&= r_{n}\bigl(s, \pi_n^{\infty}(s)\bigr)
   \;+\; \gamma(n+1) \sum_{s'} p\bigl(s' \mid s, \pi_n^{\infty}(s)\bigr)
        V_{n+1}^{\pi}(s') \\
&= Q_{n}^{\pi}\bigl(s, \pi_n^{\infty}(s)\bigr)
\end{align*}
By definition of the Bellman policy operator,
$\pi_n^{\infty}(s)$ greedily maximizes $Q_{n}^{\pi}(s,\cdot)$:
\[
Q_{n}^{\pi}(s,a) \;\le\; Q_{n}^{\pi}\bigl(s, \pi_n^{\infty}(s)\bigr)
\quad \forall\, s \in \mathcal{S},\ \forall\, a \in \mathcal{A}.
\]
In particular, for $a = \pi_n(s)$,
\[
Q_{n}^{\pi}\bigl(s, \pi_n(s)\bigr)
\;\le\; Q_{n}^{\pi}\bigl(s, \pi_n^{\infty}(s)\bigr).
\]
Combining with the identities above, we get
\[
V_{n}^{\pi^{(n-1)}}(s)
= Q_{n}^{\pi}\bigl(s, \pi_n(s)\bigr)
\;\le\; Q_{n}^{\pi}\bigl(s, \pi_n^{\infty}(s)\bigr)
= V_{n}^{\pi^{(n)}}(s)
\quad \forall\, s \in \mathcal{S}.
\]
Hence
\[
V_{n}^{\pi^{(n)}}(s) \;\ge\; V_{n}^{\pi^{(n-1)}}(s)
\quad \forall\, s \in \mathcal{S}.
\]

\textit{Step 2: Propagating improvement backward in time.}

For any $t < n$, the two policies $\pi^{(n)}$ and $\pi^{(n-1)}$ coincide
from time $t$ up to time $n-1$, and differ only at time $n$. By Lemma~2,
for each fixed $t$ and $s$, we can write
\begin{align*}
V_{t}^{\pi^{(n)}}(s)
&= \sum_{s'} C\bigl(\pi^{(n)}_{t:n-1}, s'\bigr)
    V_{n}^{\pi^{(n)}}(s') + C\bigl(\pi^{(n)}_{t:n-1}\bigr), \\
V_{t}^{\pi^{(n-1)}}(s)
&= \sum_{s'} C\bigl(\pi^{(n-1)}_{t:n-1}, s'\bigr)
    V_{n}^{\pi^{(n-1)}}(s') + C\bigl(\pi^{(n-1)}_{t:n-1}\bigr),
\end{align*}
where the coefficients depend only on the policy slice between $t$ and $n-1$.
Since $\pi^{(n)}_{t:n-1} = \pi^{(n-1)}_{t:n-1}$, these coefficients are identical:
\[
C\bigl(\pi^{(n)}_{t:n-1}, s'\bigr)
= C\bigl(\pi^{(n-1)}_{t:n-1}, s'\bigr),
\quad
C\bigl(\pi^{(n)}_{t:n-1}\bigr)
= C\bigl(\pi^{(n-1)}_{t:n-1}\bigr).
\]
Moreover, Lemma~2 guarantees that $C(\cdot,s') \ge 0$ for all $s'$.

Using $V_{n}^{\pi^{(n)}}(s') \ge V_{n}^{\pi^{(n-1)}}(s')$ for all $s'$ and the
non-negativity of the coefficients, we obtain
\begin{align*}
V_{t}^{\pi^{(n)}}(s)
&= \sum_{s'} C\bigl(\pi^{(n)}_{t:n-1}, s'\bigr)
    V_{n}^{\pi^{(n)}}(s') + C\bigl(\pi^{(n)}_{t:n-1}\bigr) \\
&\ge \sum_{s'} C\bigl(\pi^{(n-1)}_{t:n-1}, s'\bigr)
     V_{n}^{\pi^{(n-1)}}(s') + C\bigl(\pi^{(n-1)}_{t:n-1}\bigr) \\
&= V_{t}^{\pi^{(n-1)}}(s),
\end{align*}
for all $t < n$ and all $s \in \mathcal{S}$.

\textit{Step 3: Values for $t > n$}

By construction, for $t > n$, the policies $\pi^{(n)}$ and $\pi^{(n-1)}$
coincide from time $t$ onward (they only differ at time $n$), hence
\[
V_{t}^{\pi^{(n)}}(s) = V_{t}^{\pi^{(n-1)}}(s)
\quad \forall\, t > n,\ \forall\, s \in \mathcal{S}.
\]

\textit{Step 4: Monotonic improvement in $n$}

Combining the three cases $t < n$, $t = n$, and $t > n$, we conclude that
\[
V_{t}^{\pi^{(n)}}(s) \;\ge\; V_{t}^{\pi^{(n-1)}}(s)
\quad \forall\, t \in \mathbb{N},\ \forall\, s \in \mathcal{S},\ \forall\, n \ge 1.
\]
Since $\pi^{(0)} = \pi$, an induction on $n$ yields
\[
V_{t}^{\pi}(s) = V_{t}^{\pi^{(0)}}(s)
\;\le\; V_{t}^{\pi^{(n)}}(s)
\quad \forall\, t \in \mathbb{N},\ \forall\, s \in \mathcal{S},\ \forall\, n \ge 1.
\]
Finally, observe that $\pi^{(n)}$ converges pointwise to $\pi^{\infty}$ as
$n \to \infty$ (for any fixed $t$, all $\pi^{(n)}_t = \pi^{\infty}_t$ once $n \ge t$).
By continuity of the value operator in this finite-horizon setting,
\[
\lim_{n \to \infty} V_{t}^{\pi^{(n)}}(s)
= V_{t}^{\pi^{\infty}}(s)
= V_{t}^{\mathit{TP} \cdot \pi}(s).
\]
Taking limits in the inequality above gives
\[
V_{t}^{\pi}(s) \;\le\; V_{t}^{\mathit{TP} \cdot \pi}(s)
\quad \forall\, t \in \mathbb{N},\ \forall\, s \in \mathcal{S},
\]
which completes the proof.
\end{proof}

\begin{definition}[Optimal policy]
An \emph{optimal policy} $\pi^{*}$ is a (time-varying) policy
$\pi^{*} : \mathbb{N} \times \mathcal{S} \to \mathcal{A}$ such that for any
policy $\pi : \mathbb{N} \times \mathcal{S} \to \mathcal{A}$,
\[
V_{t}^{\pi^{*}}(s) \;\ge\; V_{t}^{\pi}(s)
\quad \forall\, s \in \mathcal{S},\ \forall\, t \in \mathbb{N}.
\]
\end{definition}

\begin{assumption}[(A1)]\label{asm:A1}
$\mathcal{S} \subset \mathbb{R}^{D}$ is compact.
\end{assumption}

\begin{assumption}[(A2)]\label{asm:A2}
Action space is finite, i.e.
\[
|\mathcal{A}| < \infty.
\]
\end{assumption}

\begin{assumption}[(A3)]\label{asm:A3}
Rewards are bounded, i.e.
\[
|r(s,a)| \le r_{\max}
\quad \forall (s,a) \in \mathcal{S} \times \mathcal{A}.
\]
\end{assumption}

\begin{lemma}
Under Assumption~(A3), all value functions are bounded. Specifically, for every
$\pi \in \bigcup_{t=1}^{T} \Pi_{t}$, every $t$, and every $s \in \mathcal{S}$,
\[
0 \ge V_{t}^{\pi}(s) \ge -\frac{r_{\max}}{1-\gamma}.
\]
\end{lemma}

\begin{proof}
Let $\pi \in \Pi_{n}$. Then for every $t < n$ and every $s \in \mathcal{S}$,
by definition of the value function,
\[
V_{t}^{\pi}(s)
= \mathbb{E}\Bigg[
    \sum_{i=t}^{n-1} \gamma^{i-t} r(s_{i}, a_{i})
    \,\Big|\, s_{t}=s
\Bigg].
\]
Since rewards are non-positive, we have $r(s,a) \le 0$ for all
$(s,a) \in \mathcal{S} \times \mathcal{A}$, and hence
$V_{t}^{\pi}(s) \le 0$.

Moreover,
\[
\begin{aligned}
|V_{t}^{\pi}(s)|
&\le \mathbb{E}\Bigg[
    \sum_{i=t}^{n-1} \gamma^{i-t} |r(s_{i}, a_{i})|
    \,\Big|\, s_{t}=s
\Bigg] \\
&\le \sum_{j=0}^{n-t-1} \gamma^{j} r_{\max} \\
&\le \frac{r_{\max}}{1-\gamma}.
\end{aligned}
\]
Therefore,
\[
-\frac{r_{\max}}{1-\gamma} \le V_{t}^{\pi}(s) \le 0,
\]
which proves the claim.
\end{proof}

\begin{assumption}
For any $\pi \in \Pi$ and any $t \in \mathbb{N}$, the value function
$V_{t}^{\pi}(\cdot)$ is bounded and continuous on $\mathcal{S}$, i.e.
\[
V_{t}^{\pi} \in \mathcal{C}_{b}(\mathcal{S})
\]
with respect to the supremum norm $\|\cdot\|_{\infty}$.
\end{assumption}

\begin{theorem}[Tychonoff’s Theorem]
The Cartesian product of compact topological spaces is compact. In particular,
if $\{X_i\}_{i \in I}$ is a family of compact spaces, then the product
$\prod_{i \in I} X_i$ is compact in the product topology.
\end{theorem}

\begin{lemma}
Under Assumption~1, the product $\mathcal{S} \times \mathcal{T}$ is a compact
subspace of $\mathbb{R}^{D+1}$, where $\mathcal{T} \subset \mathbb{R}$ is
bounded and closed.
\end{lemma}

\begin{proof}
By Assumption~1, $\mathcal{S} \subset \mathbb{R}^{D}$ is compact. Since
$\mathcal{T} \subset \mathbb{R}$ is bounded and closed, it is also compact
(e.g.\ by the Heine--Borel theorem). Therefore, both $\mathcal{S}$ and
$\mathcal{T}$ are compact spaces.

By Tychonoff’s theorem (applied to the finite product of compact spaces),
the Cartesian product $\mathcal{S} \times \mathcal{T}$ is compact in the
product topology. Since we can view $\mathcal{S} \times \mathcal{T}$ as a
subset of $\mathbb{R}^{D+1}$ with the subspace topology, it is a compact
subspace of $\mathbb{R}^{D+1}$.
\end{proof}

\begin{lemma}
Under Assumption~1, all time-varying value functions lie in a complete metric
space $(\hat{\mathcal{V}}, d)$, where the metric $d$ is defined by
\[
d(V^{1}, V^{2})
:= \| V^{1} - V^{2} \|_{\infty}
= \max_{t \in \mathcal{T}} \max_{s \in \mathcal{S}}
    \bigl| V^{1}_{t}(s) - V^{2}_{t}(s) \bigr|
\quad \forall\, V^{1}, V^{2} \in \hat{\mathcal{V}}.
\]
\end{lemma}

\begin{proof}
Consider the product space $\mathcal{S} \times \mathcal{T}$, which is compact
in $\mathbb{R}^{D+1}$ by Lemma~3. Let $\mathcal{B}(\mathcal{S} \times \mathcal{T})$
denote the set of all bounded real-valued functions on $\mathcal{S} \times \mathcal{T}$,
equipped with the supremum norm
\[
\| f \|_{\infty}
:= \sup_{(s,t) \in \mathcal{S} \times \mathcal{T}} |f(s,t)|.
\]
It is well known that $\mathcal{B}(\mathcal{S} \times \mathcal{T})$ is a Banach
space under $\|\cdot\|_{\infty}$, hence a complete metric space with respect to
the induced metric.

Each time-varying value function $\hat{v}$ can be identified with a bounded
function on $\mathcal{S} \times \mathcal{T}$ via
\[
\hat{v}(s,t) := V^{\pi}_{t}(s).
\]
Let $\hat{\mathcal{V}}$ denote the collection of all such time-varying value
functions. Then
\[
\hat{\mathcal{V}} \subset \mathcal{B}(\mathcal{S} \times \mathcal{T}),
\]
and the metric $d$ defined by
\[
d(V^{1}, V^{2})
= \| V^{1} - V^{2} \|_{\infty}
\]
is exactly the restriction of the supremum norm metric on
$\mathcal{B}(\mathcal{S} \times \mathcal{T})$ to the subspace $\hat{\mathcal{V}}$.

Since any subspace of a complete metric space is complete with respect to the
induced metric, it follows that $(\hat{\mathcal{V}}, d)$ is complete. This
concludes the proof.
\end{proof}

\begin{definition}[Restriction operator]\label{def:restriction}
Let $\hat{\mathcal{V}}$ denote the space of bounded time-varying value
functions on $\mathcal{S} \times \mathcal{T}$, and let
$\mathcal{V}$ denote the space of value functions indexed on discrete times
$t \in \mathbb{N}$.

Define the map
\[
\mathcal{R} : \hat{\mathcal{V}} \to \mathcal{V}
\]
by
\[
\bigl(\mathcal{R}(\hat{v})\bigr)_{t}(s)
:= \hat{v}(s,t),
\quad \forall\, s \in \mathcal{S},\ \forall\, t \in \mathbb{N},
\]
for any $\hat{v} \in \hat{\mathcal{V}}$.
\end{definition}

\begin{lemma}\label{lem:R-continuous}
Let $\{\hat{v}^{n}\}_{n=1}^{\infty} \subset \hat{\mathcal{V}}$ be a sequence
such that $\hat{v}^{n} \to \hat{v}$ in $(\hat{\mathcal{V}}, d)$, i.e.
\[
\lim_{n \to \infty} d(\hat{v}^{n}, \hat{v}) = 0,
\]
where
\[
d(\hat{v}^{1}, \hat{v}^{2})
:= \|\hat{v}^{1} - \hat{v}^{2}\|_{\infty}
= \max_{(s,t) \in \mathcal{S} \times \mathcal{T}}
    \bigl| \hat{v}^{1}(s,t) - \hat{v}^{2}(s,t) \bigr|.
\]
Then
\[
\lim_{n \to \infty} \mathcal{R}(\hat{v}^{n})
= \mathcal{R}(\hat{v})
\]
in $\mathcal{V}$ with respect to the induced supremum metric.
\end{lemma}

\begin{proof}
By definition of convergence in $(\hat{\mathcal{V}}, d)$, for any
$\varepsilon > 0$, there exists $N_{0} \in \mathbb{N}$ such that for all
$n \ge N_{0}$,
\[
d(\hat{v}^{n}, \hat{v})
= \max_{(s,t) \in \mathcal{S} \times \mathcal{T}}
    \bigl| \hat{v}^{n}(s,t) - \hat{v}(s,t) \bigr|
\le \varepsilon.
\]
In particular, this bound holds when we restrict $(s,t)$ to the subset
$\mathcal{S} \times \mathbb{N} \subset \mathcal{S} \times \mathcal{T}$, so
\[
\max_{s \in \mathcal{S},\, t \in \mathbb{N}}
    \bigl| \hat{v}^{n}(s,t) - \hat{v}(s,t) \bigr|
\le \varepsilon.
\]
But by Definition~\ref{def:restriction}, this is exactly
\[
\max_{s \in \mathcal{S},\, t \in \mathbb{N}}
    \bigl| (\mathcal{R}(\hat{v}^{n}))_{t}(s)
          - (\mathcal{R}(\hat{v}))_{t}(s) \bigr|
= d\bigl(\mathcal{R}(\hat{v}^{n}), \mathcal{R}(\hat{v})\bigr).
\]
Hence, for all $n \ge N_{0}$,
\[
d\bigl(\mathcal{R}(\hat{v}^{n}), \mathcal{R}(\hat{v})\bigr)
\le \varepsilon,
\]
which shows $\mathcal{R}(\hat{v}^{n}) \to \mathcal{R}(\hat{v})$ in
$\mathcal{V}$ as $n \to \infty$.
\end{proof}

\begin{definition}[Bellman value operator]\label{def:bellman-operator}
The \emph{Bellman value operator} $T : \hat{\mathcal{V}} \to \hat{\mathcal{V}}$
is defined for any $V \in \hat{\mathcal{V}}$ by
\[
(TV)_{t}(s)
:= \max_{a \in \mathcal{A}}
\Bigl[
    r_{t}(s,a)
    + \gamma(t+1) \sum_{s'} p(s' \mid s,a)\, V_{t+1}(s')
\Bigr],
\quad \forall\, s \in \mathcal{S},\ t \in \mathbb{N}.
\]
\end{definition}

\begin{lemma}\label{lem:T-R-commute}
The Bellman value operator $T$ commutes with the restriction operator
$\mathcal{R}$ in the sense that for all $V \in \hat{\mathcal{V}}$,
\[
\bigl(\mathcal{R}(TV)\bigr)_{t}(s)
= \bigl(T\,\mathcal{R}(V)\bigr)_{t}(s)
\quad \forall\, s \in \mathcal{S},\ \forall\, t \in \mathbb{N}.
\]
\end{lemma}

\begin{proof}
Fix any $V \in \hat{\mathcal{V}}$, $s \in \mathcal{S}$ and $t \in \mathbb{N}$.
By Definition~\ref{def:bellman-operator},
\[
(TV)_{t}(s)
= \max_{a \in \mathcal{A}}
\Bigl[
    r_{t}(s,a)
    + \gamma(t+1) \sum_{s'} p(s' \mid s,a)\, V_{t+1}(s')
\Bigr].
\]
Applying the restriction operator $\mathcal{R}$ simply reads off these values
at integer times, so
\[
\bigl(\mathcal{R}(TV)\bigr)_{t}(s)
= (TV)_{t}(s)
= \max_{a \in \mathcal{A}}
\Bigl[
    r_{t}(s,a)
    + \gamma(t+1) \sum_{s'} p(s' \mid s,a)\, V_{t+1}(s')
\Bigr].
\]

On the other hand, for the restricted function $\mathcal{R}(V)$ we have
\[
\bigl(\mathcal{R}(V)\bigr)_{t+1}(s')
= V_{t+1}(s')
\quad \forall\, s' \in \mathcal{S},
\]
since $\mathcal{R}$ preserves the values of $V$ at integer time points.
Therefore, applying $T$ to $\mathcal{R}(V)$ yields
\begin{align*}
\bigl(T\,\mathcal{R}(V)\bigr)_{t}(s)
&= \max_{a \in \mathcal{A}}
\Bigl[
    r_{t}(s,a)
    + \gamma(t+1) \sum_{s'} p(s' \mid s,a)\,
        \bigl(\mathcal{R}(V)\bigr)_{t+1}(s')
\Bigr] \\
&= \max_{a \in \mathcal{A}}
\Bigl[
    r_{t}(s,a)
    + \gamma(t+1) \sum_{s'} p(s' \mid s,a)\, V_{t+1}(s')
\Bigr].
\end{align*}
Comparing the two expressions, we see that for all $s \in \mathcal{S}$ and
$t \in \mathbb{N}$,
\[
\bigl(\mathcal{R}(TV)\bigr)_{t}(s)
= \bigl(T\,\mathcal{R}(V)\bigr)_{t}(s),
\]
which proves that $T$ and $\mathcal{R}$ commute on integer time indices.
\end{proof}

\begin{lemma}[Bellman value operator increases value]
Let $V \in \hat{\mathcal{V}}$ be the value function of some deterministic
policy $\pi$. Then the Bellman value operator $T$ satisfies
\[
(TV)_{t}(s) \;\ge\; V_{t}(s)
\quad \forall\, s \in \mathcal{S},\ \forall\, t \in \mathbb{N}.
\]
\end{lemma}

\begin{proof}
Fix any $s \in \mathcal{S}$ and $t \in \mathbb{N}$. By definition of the
Bellman value operator,
\begin{align*}
(TV)_{t}(s)
&= \max_{a \in \mathcal{A}}
    \Bigl[
        r_{t}(s,a)
        + \gamma(t+1) \sum_{s'} p(s' \mid s,a)\, V_{t+1}(s')
    \Bigr].
\end{align*}
In particular, this maximum is at least as large as the value obtained by
choosing the policy action $a = \pi_{t}(s)$:
\begin{align*}
(TV)_{t}(s)
&\ge
    r_{t}\bigl(s,\pi_{t}(s)\bigr)
    + \gamma(t+1) \sum_{s'} p\bigl(s' \mid s,\pi_{t}(s)\bigr)\, V_{t+1}(s').
\end{align*}
Since $V$ is the value function of $\pi$, it satisfies the (dynamic) Bellman
equation for $\pi$:
\[
V_{t}(s)
= r_{t}\bigl(s,\pi_{t}(s)\bigr)
  + \gamma(t+1) \sum_{s'} p\bigl(s' \mid s,\pi_{t}(s)\bigr)\, V_{t+1}(s').
\]
Combining the two displays gives
\[
(TV)_{t}(s) \;\ge\; V_{t}(s)
\quad \forall\, s \in \mathcal{S},\ \forall\, t \in \mathbb{N},
\]
which proves the claim.
\end{proof}

\begin{theorem}[Extreme Value Theorem]
Let $K$ be a non-empty compact subset of $\mathbb{R}^{n}$ and let
$f : K \to \mathbb{R}$ be continuous. Then $f$ is bounded on $K$ and there
exists $p \in K$ such that
\[
f(p) = \sup_{x \in K} f(x).
\]
\end{theorem}

\begin{lemma}\label{lem:max-attained}
For any $f \in \hat{\mathcal{V}}$, there exists $x^{*} \in \mathcal{S} \times \mathcal{T}$
such that
\[
f(x^{*}) = \sup_{x \in \mathcal{S} \times \mathcal{T}} f(x).
\]
\end{lemma}

\begin{proof}
By Assumption~1 and Lemma~3, the set
$\mathcal{S} \times \mathcal{T} \subset \mathbb{R}^{D+1}$ is compact. By
definition of $\hat{\mathcal{V}}$, each $f \in \hat{\mathcal{V}}$ is a bounded
continuous function on $\mathcal{S} \times \mathcal{T}$. Therefore, by the
Extreme Value Theorem, $f$ attains its supremum on $\mathcal{S} \times \mathcal{T}$,
i.e.\ there exists $x^{*} \in \mathcal{S} \times \mathcal{T}$ such that
\[
f(x^{*}) = \sup_{x \in \mathcal{S} \times \mathcal{T}} f(x).
\]
This proves the claim.
\end{proof}

\begin{lemma}\label{lem:max-lipschitz}
The max operator is Lipschitz on $\hat{\mathcal{V}}$ with respect to the
supremum norm. More precisely, for any $f,g \in \hat{\mathcal{V}}$,
\[
\bigl|\max_{x} f(x) - \max_{x} g(x)\bigr|
\;\le\; \max_{x} |f(x) - g(x)|.
\]
\end{lemma}

\begin{proof}
Let $M_f := \max_{x} f(x)$ and $M_g := \max_{x} g(x)$, where the maxima are
attained by Lemma~\ref{lem:max-attained}. Let $a^{*} \in \arg\max_{x} f(x)$
and $b^{*} \in \arg\max_{x} g(x)$, so
\[
M_f = f(a^{*}), \quad M_g = g(b^{*}).
\]

First bound $M_f - M_g$ from above:
\begin{align*}
M_f - M_g
&= f(a^{*}) - \max_{x} g(x) \\
&\le f(a^{*}) - g(a^{*}) \\
&\le \max_{x} |f(x) - g(x)|.
\end{align*}
The first inequality follows since $\max_{x} g(x) \ge g(a^{*})$.

Similarly, swapping the roles of $f$ and $g$ gives
\begin{align*}
M_g - M_f
&= g(b^{*}) - \max_{x} f(x) \\
&\le g(b^{*}) - f(b^{*}) \\
&\le \max_{x} |f(x) - g(x)|.
\end{align*}

Combining both bounds, we obtain
\[
|M_f - M_g|
= \bigl|\max_{x} f(x) - \max_{x} g(x)\bigr|
\le \max_{x} |f(x) - g(x)|,
\]
which completes the proof.
\end{proof}

\begin{lemma}\label{lem:T-contraction}
Suppose the (possibly time-varying) discount satisfies $\gamma \in [0,1)$, then the (general) Bellman value operator $T$ is a contraction on the metric space $(\hat{\mathcal{V}}, d)$, i.e.
\[
d(TV^{1}, TV^{2}) \;\le\; \gamma \, d(V^{1}, V^{2})
\quad \forall\, V^{1}, V^{2} \in \hat{\mathcal{V}},
\]
where
\[
d(V^{1}, V^{2})
:= \max_{t \in \mathbb{N}} \max_{s \in \mathcal{S}}
   \bigl|V^{1}_{t}(s) - V^{2}_{t}(s)\bigr|.
\]
\end{lemma}

\begin{proof}
Fix arbitrary $V^{1}, V^{2} \in \hat{\mathcal{V}}$, and consider for any
$(s,t) \in \mathcal{S} \times \mathbb{N}$ the difference
\[
\bigl|(TV^{1})_{t}(s) - (TV^{2})_{t}(s)\bigr|.
\]
By definition of the Bellman value operator,
\begin{align*}
(TV^{1})_{t}(s)
&= \max_{a}
    \Bigl[
        r_{t}(s,a)
        + \gamma \sum_{s'} p(s' \mid s,a) V^{1}_{t+1}(s')
    \Bigr], \\
(TV^{2})_{t}(s)
&= \max_{a}
    \Bigl[
        r_{t}(s,a)
        + \gamma \sum_{s'} p(s' \mid s,a) V^{2}_{t+1}(s')
    \Bigr].
\end{align*}
Define, for each $a$,
\[
F^{1}_{t}(s,a)
:= r_{t}(s,a)
   + \gamma \sum_{s'} p(s' \mid s,a) V^{1}_{t+1}(s'),
\quad
F^{2}_{t}(s,a)
:= r_{t}(s,a)
   + \gamma \sum_{s'} p(s' \mid s,a) V^{2}_{t+1}(s').
\]
Then
\[
(TV^{1})_{t}(s) = \max_{a} F^{1}_{t}(s,a),
\quad
(TV^{2})_{t}(s) = \max_{a} F^{2}_{t}(s,a).
\]
Using the Lipschitz property of the max operator (Lemma~\ref{lem:max-lipschitz}),
we obtain
\begin{align*}
\bigl|(TV^{1})_{t}(s) - (TV^{2})_{t}(s)\bigr|
&= \bigl|\max_{a} F^{1}_{t}(s,a) - \max_{a} F^{2}_{t}(s,a)\bigr| \\
&\le \max_{a} \bigl|F^{1}_{t}(s,a) - F^{2}_{t}(s,a)\bigr|.
\end{align*}
For each $a$,
\begin{align*}
\bigl|F^{1}_{t}(s,a) - F^{2}_{t}(s,a)\bigr|
&= \gamma \Bigl|
    \sum_{s'} p(s' \mid s,a)
        \bigl(V^{1}_{t+1}(s') - V^{2}_{t+1}(s')\bigr)
   \Bigr| \\
&\le \gamma
    \sum_{s'} p(s' \mid s,a)
        \bigl|V^{1}_{t+1}(s') - V^{2}_{t+1}(s')\bigr| \\
&\le \gamma
    \max_{s'} \bigl|V^{1}_{t+1}(s') - V^{2}_{t+1}(s')\bigr| \\
&\le \gamma \, d(V^{1}, V^{2}),
\end{align*}
where we used the fact that $p(\cdot \mid s,a)$ is a probability distribution
and the definition of $d$.

Taking the maximum over $a$ gives
\[
\bigl|(TV^{1})_{t}(s) - (TV^{2})_{t}(s)\bigr|
\le \gamma\, d(V^{1}, V^{2}).
\]
Finally, taking the supremum over all $s \in \mathcal{S}$ and $t \in \mathbb{N}$,
and using $\gamma(t+1) \le \gamma$, we obtain
\begin{align*}
d(TV^{1}, TV^{2})
&= \max_{t,s} \bigl|(TV^{1})_{t}(s) - (TV^{2})_{t}(s)\bigr| \\
&\le \max_{t} \gamma \, d(V^{1}, V^{2}) \\
&\le \gamma \, d(V^{1}, V^{2}).
\end{align*}
Thus $T$ is a contraction mapping on $(\hat{\mathcal{V}}, d)$ with modulus at
most $\gamma$.
\end{proof}


\begin{lemma}\label{lem:V-extension}
For every $V \in \mathcal{V}$, there exists $\hat{V} \in \hat{\mathcal{V}}$ such that
\[
V(s,t) = (\mathcal{R}\hat{V})_{t}(s)
\quad \forall\, s \in \mathcal{S},\ \forall\, t \in \mathbb{N}_{\le T}.
\]
\end{lemma}

\begin{proof}
We prove the claim by explicit construction. We need to construct a bounded,
continuous function
\[
\hat{V} : \mathcal{S} \times \mathcal{T} \to \mathbb{R}
\]
such that its restriction to integer time points coincides with $V$, i.e.
$(\mathcal{R}\hat{V})_{t}(s) = \hat{V}(s,t) = V(s,t)$ for all
$(s,t) \in \mathcal{S} \times \mathbb{N}_{\le T}$.

Define
\[
\hat{V}(s,t)
:= \sum_{k=0}^{T-1} \phi_{k}(t)\, V(s,k),
\quad \forall (s,t) \in \mathcal{S} \times \mathcal{T},
\]
where
\[
\phi_{k}(t)
:= 
\begin{cases}
1 - |t - k|, & \text{if } |t - k| < 1, \\
0,           & \text{otherwise}.
\end{cases}
\]
This construction linearly interpolates between the discrete time points.

Observe that for integer $t \in \{0,\dots,T-1\}$ we have
\[
\phi_{k}(t)
=
\begin{cases}
1, & k = t, \\
0, & k \neq t,
\end{cases}
\]
hence
\[
\hat{V}(s,t)
= \sum_{k=0}^{T-1} \phi_{k}(t)\, V(s,k)
= V(s,t),
\quad \forall (s,t) \in \mathcal{S} \times \mathbb{N}_{\le T}.
\]
Therefore $(\mathcal{R}\hat{V})_{t}(s) = \hat{V}(s,t) = V(s,t)$ on the discrete
time grid, as required.

\medskip
\noindent\textbf{Boundedness.}
Since $V \in \mathcal{V} \subset \mathcal{C}_{b}(\mathcal{S} \times \mathbb{N})$,
there exists $M < \infty$ such that
\[
\|V(\cdot,k)\|_{\infty}
:= \sup_{s \in \mathcal{S}} |V(s,k)|
\le M,
\quad \forall\, k \in \{0,1,\dots,T-1\}.
\]
For each fixed $t \in \mathcal{T}$, the function $\phi_{k}(t)$ satisfies
$0 \le \phi_{k}(t) \le 1$, and $\phi_{k}(t) \neq 0$ only if $|t-k| < 1$.
Since $k$ is integer, there are at most two indices $k$ such that
$\phi_{k}(t) \neq 0$.

Thus, for any $(s,t) \in \mathcal{S} \times \mathcal{T}$,
\begin{align*}
|\hat{V}(s,t)|
&= \Bigl|\sum_{k=0}^{T-1} \phi_{k}(t)\, V(s,k)\Bigr| \\
&\le \sum_{k=0}^{T-1} \phi_{k}(t)\, |V(s,k)| \\
&\le \sum_{k=0}^{T-1} \phi_{k}(t)\, M
\;\le\; 2M,
\end{align*}
since at most two terms in the sum are non-zero and each $\phi_{k}(t) \le 1$.
Hence
\[
\|\hat{V}\|_{\infty}
:= \sup_{(s,t) \in \mathcal{S} \times \mathcal{T}} |\hat{V}(s,t)|
\le 2M < \infty,
\]
so $\hat{V}$ is bounded.

\medskip
\noindent\textbf{Continuity.}
For each fixed $k$, the map $s \mapsto V(s,k)$ is continuous on $\mathcal{S}$
(by assumption on $\mathcal{V}$), and $t \mapsto \phi_{k}(t)$ is continuous on
$\mathcal{T}$. Therefore the product
\[
(s,t) \mapsto \phi_{k}(t)\, V(s,k)
\]
is continuous on $\mathcal{S} \times \mathcal{T}$. A finite sum of continuous
functions is continuous, so $\hat{V}$ is continuous on $\mathcal{S} \times \mathcal{T}$.

\medskip

We have thus constructed $\hat{V} \in \mathcal{C}_{b}(\mathcal{S} \times \mathcal{T})$,
hence $\hat{V} \in \hat{\mathcal{V}}$, such that $\mathcal{R}\hat{V} = V$.
This concludes the proof.
\end{proof}

\begin{definition}[Contraction mapping]
Let $(\mathcal{X}, d)$ be a metric space. A map $T : \mathcal{X} \to \mathcal{X}$
is called a \emph{contraction mapping} on $\mathcal{X}$ if there exists a
constant $q \in [0,1)$ such that
\[
d\bigl(T(x), T(y)\bigr)
\;\le\; q \, d(x,y)
\quad \forall\, x,y \in \mathcal{X}.
\]
\end{definition}

\begin{theorem}[Banach Fixed Point Theorem]
Let $(\mathcal{X}, d)$ be a non-empty complete metric space, and let
$T : \mathcal{X} \to \mathcal{X}$ be a contraction mapping. Then:
\begin{enumerate}
    \item There exists a unique fixed point $x^{*} \in \mathcal{X}$ such that
    \[
    T(x^{*}) = x^{*}.
    \]
    \item Moreover, for any initial point $x_{0} \in \mathcal{X}$, the sequence
    $(x_{n})_{n \in \mathbb{N}}$ defined recursively by
    \[
    x_{n} := T(x_{n-1}), \quad n \ge 1,
    \]
    converges to $x^{*}$, i.e.
    \[
    \lim_{n \to \infty} x_{n} = x^{*}.
    \]
\end{enumerate}
\end{theorem}

\begin{theorem}[Existence and uniqueness of optimal value function]\label{thm:optimal-V}
There exists a unique optimal value function $V^{*} \in \mathcal{V}$ such that
it is invariant under the Bellman value operator $T : \hat{\mathcal{V}} \to \hat{\mathcal{V}}$, i.e.
\[
V^{*}_{t}(s) = (TV^{*})_{t}(s)
\quad \forall\, s \in \mathcal{S},\ \forall\, t \in \mathcal{T}.
\]
\end{theorem}

\begin{proof}
\textbf{Step 1: Fixed point of $T$ in $\hat{\mathcal{V}}$.}
By Lemma~\ref{lem:T-contraction}, the Bellman value operator
$T : \hat{\mathcal{V}} \to \hat{\mathcal{V}}$ is a contraction mapping on the
metric space $(\hat{\mathcal{V}}, d)$. By Lemma~4, $(\hat{\mathcal{V}}, d)$
is complete. Hence, by the Banach Fixed Point Theorem, there exists a unique
$\hat{V}^{*} \in \hat{\mathcal{V}}$ such that
\[
T\hat{V}^{*} = \hat{V}^{*},
\]
and, moreover, for any initial $\hat{V} \in \hat{\mathcal{V}}$,
\[
\hat{V}^{*}(s,t)
= \lim_{n \to \infty} (T^{n}\hat{V})(s,t)
\quad \forall\, s \in \mathcal{S},\ \forall\, t \in \mathcal{T}.
\]
In particular, the fixed-point condition can be written as
\[
\hat{V}^{*}_{t}(s) = (T\hat{V}^{*})_{t}(s)
\quad \forall\, s \in \mathcal{S},\ \forall\, t \in \mathcal{T}.
\]

\medskip
\noindent\textbf{Step 2: Optimality and uniqueness in $\hat{\mathcal{V}}$.}
By Lemma~7, the Bellman value operator $T$ is value-improving in the sense
that, when applied to the value function of a policy, it yields a value
function that is pointwise no worse. Iterating $T$ starting from any such
value function and using convergence to $\hat{V}^{*}$ implies that
$\hat{V}^{*}$ is pointwise greater than or equal to any other value function
in $\hat{\mathcal{V}}$:
\[
\hat{V}^{*}(s,t) \;\ge\; \hat{V}(s,t)
\quad \forall\, s \in \mathcal{S},\ \forall\, t \in \mathcal{T},\ \forall\, \hat{V} \in \hat{\mathcal{V}}.
\]
By Definition~4 (optimality), $\hat{V}^{*}$ is thus the optimal value function
in $\hat{\mathcal{V}}$.

To see uniqueness, suppose there were another optimal value function
$\hat{V}' \in \hat{\mathcal{V}}$. Optimality would then imply
\[
\hat{V}^{*}(s,t) \;\ge\; \hat{V}'(s,t)
\quad\text{and}\quad
\hat{V}'(s,t) \;\ge\; \hat{V}^{*}(s,t)
\quad \forall\, s \in \mathcal{S},\ \forall\, t \in \mathcal{T},
\]
hence
\[
\hat{V}^{*}(s,t) = \hat{V}'(s,t)
\quad \forall\, s \in \mathcal{S},\ \forall\, t \in \mathcal{T},
\]
so the optimal value function in $\hat{\mathcal{V}}$ is unique.

\medskip
\noindent\textbf{Step 3: Induced optimal value function in $\mathcal{V}$.}
Define
\[
V^{*}(s,t) := \bigl(\mathcal{R}\hat{V}^{*}\bigr)_{t}(s),
\quad \forall\, (s,t) \in \mathcal{S} \times \mathbb{N}_{\le T}.
\]
By Lemma~\ref{lem:V-extension}, for any $V \in \mathcal{V}$ there exists
$\hat{V} \in \hat{\mathcal{V}}$ such that $\mathcal{R}\hat{V} = V$. We can
denote one such extension by $\hat{V} = \mathcal{R}^{-1}V$ for notational
convenience (any choice of extension suffices for the argument).

Since $\hat{V}^{*}$ is optimal in $\hat{\mathcal{V}}$, we have
\[
\hat{V}^{*}(s,t) \;\ge\; \hat{V}(s,t)
= (\mathcal{R}^{-1}V)(s,t)
\quad \forall\, (s,t) \in \mathcal{S} \times \mathcal{T},
\ \forall\, V \in \mathcal{V}.
\]
Applying $\mathcal{R}$ to both sides and using $\mathcal{R}\hat{V} = V$ yields
\[
V^{*}(s,t)
= \bigl(\mathcal{R}\hat{V}^{*}\bigr)_{t}(s)
\;\ge\; \bigl(\mathcal{R}\mathcal{R}^{-1}V\bigr)_{t}(s)
= V_{t}(s),
\quad \forall\, (s,t) \in \mathcal{S} \times \mathbb{N}_{\le T},\ \forall\, V \in \mathcal{V}.
\]
Thus $V^{*}$ is optimal in $\mathcal{V}$.

\medskip
\noindent\textbf{Step 4: Constructing $V^{*}$ via value iteration.}
We know that
\[
\hat{V}^{*}(s,t) = \lim_{n \to \infty} (T^{n}\hat{V})(s,t)
\quad \forall\, \hat{V} \in \hat{\mathcal{V}}.
\]
By Lemma~\ref{lem:R-continuous}, the restriction operator $\mathcal{R}$ is
continuous, so
\[
V^{*}(s,t)
= \bigl(\mathcal{R}\hat{V}^{*}\bigr)_{t}(s)
= \lim_{n \to \infty} \bigl(\mathcal{R}T^{n}\hat{V}\bigr)_{t}(s).
\]
By Lemma~\ref{lem:T-R-commute}, $\mathcal{R}$ and $T$ commute, hence
\[
\mathcal{R}(T^{n}\hat{V})
= T^{n}(\mathcal{R}\hat{V})
= T^{n}V,
\]
where $V = \mathcal{R}\hat{V} \in \mathcal{V}$. Therefore
\[
V^{*}(s,t)
= \lim_{n \to \infty} \bigl(T^{n}V\bigr)_{t}(s)
\quad \forall\, s \in \mathcal{S},\ \forall\, t \in \mathbb{N}_{\le T},\ \forall\, V \in \mathcal{V}.
\]
This shows that the optimal value function $V^{*}$ can be obtained by
iteratively applying the Bellman value operator starting from any initial
$V \in \mathcal{V}$.

Uniqueness of $V^{*}$ in $\mathcal{V}$ follows immediately: if $V'$ were
another optimal value function, then its extension
$\hat{V}' = \mathcal{R}^{-1}V'$ would be an optimal value function in
$\hat{\mathcal{V}}$, and by uniqueness in $\hat{\mathcal{V}}$ we must have
$\hat{V}' = \hat{V}^{*}$, hence $V' = V^{*}$.

\medskip
\noindent\textbf{Step 5: Bellman fixed point relation in $\mathcal{V}$.}
Finally, using the commutativity of $T$ and $\mathcal{R}$, we have
\[
V^{*}_{t}(s)
= \bigl(\mathcal{R}\hat{V}^{*}\bigr)_{t}(s)
= \bigl(\mathcal{R}T\hat{V}^{*}\bigr)_{t}(s)
= \bigl(T\,\mathcal{R}\hat{V}^{*}\bigr)_{t}(s)
= (TV^{*})_{t}(s),
\]
for all $s \in \mathcal{S}$ and $t \in \mathcal{T}$. This shows that $V^{*}$
is invariant under $T$ in $\mathcal{V}$ and completes the proof.
\end{proof}


\begin{definition}[Bellman policy operator]\label{def:TP}
The \emph{Bellman policy operator} $\mathit{TP} : \Pi \to \Pi$ is defined for
any policy $\pi \in \Pi$, and for all $s \in \mathcal{S}$ and $t \in \mathbb{N}$, by
\[
(\mathit{TP} \cdot \pi)_{t}(s)
:= \arg\max_{a \in \mathcal{A}} Q^{\pi}_{t}(s,a)
= \arg\max_{a \in \mathcal{A}}
    \Bigl[
        r_{t}(s,a)
        + \gamma(t+1) \sum_{s'} p(s' \mid s,a)\, V^{\pi}_{t+1}(s')
    \Bigr].
\]
\end{definition}

\begin{theorem}[Bellman policy operator improves value]\label{thm:policy-improvement}
Let $\pi \in \Pi$ be any (time-varying) policy and let $\mathit{TP} \cdot \pi$
be the Bellman policy update as in Definition~\ref{def:TP}. Then, for all
states and times,
\[
V^{\pi}_{t}(s) \;\le\; V^{\mathit{TP} \cdot \pi}_{t}(s)
\quad \forall\, s \in \mathcal{S},\ \forall\, t \in \mathbb{N}.
\]
\end{theorem}

\begin{proof}
We start from the Bellman value operator applied to $V^{\pi}$. By definition,
\[
(TV^{\pi})_{t}(s)
= \max_{a} Q^{\pi}_{t}(s,a)
= Q^{\pi}_{t}\bigl(s, (\mathit{TP} \cdot \pi)_{t}(s)\bigr),
\]
where the second equality follows from the definition of the Bellman policy
operator (Definition~\ref{def:TP}). Thus, to show
$V^{\pi}_{t}(s) \le V^{\mathit{TP} \cdot \pi}_{t}(s)$, it suffices to prove
the stronger inequality
\[
Q^{\pi}_{t}\bigl(s, (\mathit{TP} \cdot \pi)_{t}(s)\bigr)
\;\le\;
Q^{\mathit{TP} \cdot \pi}_{t}\bigl(s, (\mathit{TP} \cdot \pi)_{t}(s)\bigr)
\quad \forall\, s \in \mathcal{S},\ \forall\, t \in \mathbb{N}.
\]

\medskip
\noindent\textbf{Step 1: Hybrid policies.}
Define, for each $n \in \mathbb{N}$, a \emph{hybrid} policy
\[
\pi^{(n)}
:= \bigl[\,(\mathit{TP} \cdot \pi)_{1}, \dots, (\mathit{TP} \cdot \pi)_{n},
            \pi_{n+1}, \pi_{n+2}, \dots \bigr].
\]
Thus $\pi^{(0)} = \pi$ and $\pi^{(\infty)} := \lim_{n \to \infty} \pi^{(n)}$
is exactly the fully improved policy $\mathit{TP} \cdot \pi$.

We compare $Q^{\pi}_{t}$ and $Q^{\pi^{(n)}}_{t}$ for different $t$ and $n$.

\medskip
\noindent\textbf{Step 2: Improvement at time $t = n$.}
Fix $n \in \mathbb{N}$ and $s \in \mathcal{S}$. By the definition of $Q^{\pi}$,
\begin{align*}
Q_{n}^{\pi}\bigl(s, \pi_{n}(s)\bigr)
&\le \max_{a}
    \Bigl[
        r_{n}(s,a)
        + \gamma(n+1) \sum_{s'} p(s' \mid s,a)\, V_{n+1}^{\pi}(s')
    \Bigr] \\
&= r_{n}\bigl(s, (\mathit{TP} \cdot \pi)_{n}(s)\bigr)
   + \gamma(n+1) \sum_{s'} p\bigl(s' \mid s, (\mathit{TP} \cdot \pi)_{n}(s)\bigr)
        V_{n+1}^{\pi}(s'),
\end{align*}
where the inequality comes from the definition of the max, and the equality
uses the Bellman policy operator’s choice at time $n$:
$(\mathit{TP} \cdot \pi)_{n}(s) \in \arg\max_{a} Q_{n}^{\pi}(s,a)$.

By construction of $\pi^{(n)}$, we have
\[
\pi^{(n)}_{n} = (\mathit{TP} \cdot \pi)_{n}, \qquad
\pi^{(n)}_{k} = \pi_{k} \text{ for all } k \ge n+1.
\]
Hence the rollout from time $n+1$ onward under $\pi^{(n)}$ is identical to
that under $\pi$, and
\[
V^{\pi^{(n)}}_{n+1}(s') = V^{\pi}_{n+1}(s') \quad \forall\, s' \in \mathcal{S}.
\]
Using the Bellman equation for $Q^{\pi^{(n)}}$,
\begin{align*}
Q_{n}^{\pi^{(n)}}\bigl(s, \pi^{(n)}_{n}(s)\bigr)
&= r_{n}\bigl(s, \pi^{(n)}_{n}(s)\bigr)
   + \gamma(n+1) \sum_{s'} p\bigl(s' \mid s,\pi^{(n)}_{n}(s)\bigr)
        V_{n+1}^{\pi^{(n)}}(s') \\
&= r_{n}\bigl(s, (\mathit{TP} \cdot \pi)_{n}(s)\bigr)
   + \gamma(n+1) \sum_{s'} p\bigl(s' \mid s,(\mathit{TP} \cdot \pi)_{n}(s)\bigr)
        V_{n+1}^{\pi}(s'),
\end{align*}
which matches the right-hand side above. Therefore,
\[
Q_{n}^{\pi}\bigl(s, \pi_{n}(s)\bigr)
\;\le\; Q_{n}^{\pi^{(n)}}\bigl(s, \pi^{(n)}_{n}(s)\bigr)
\quad \forall\, s \in \mathcal{S}.
\]

\medskip
\noindent\textbf{Step 3: Times $t > n$.}
For $t > n$, by definition $\pi^{(n)}_{t} = \pi_{t}$ and
$\pi^{(n)}_{k} = \pi_{k}$ for all $k \ge t$. Hence the policy from time $t$
onward is identical for $\pi$ and $\pi^{(n)}$, which implies
\[
Q_{t}^{\pi}\bigl(s, \pi_{t}(s)\bigr)
= Q_{t}^{\pi^{(n)}}\bigl(s, \pi^{(n)}_{t}(s)\bigr)
\quad \forall\, s \in \mathcal{S},\ \forall\, t > n.
\]

\medskip
\noindent\textbf{Step 4: Times $t < n$ by backward induction.}
Consider $t = n-1$. Using the Bellman equation and the greedy choice at time
$n-1$, we have
\begin{align*}
Q_{n-1}^{\pi}\bigl(s, \pi_{n-1}(s)\bigr)
&\le \max_{a}
    \Bigl[
        r_{n-1}(s,a)
        + \gamma(n) \sum_{s'} p(s' \mid s,a)\, V_{n}^{\pi}(s')
    \Bigr] \\
&= r_{n-1}\bigl(s, (\mathit{TP} \cdot \pi)_{n-1}(s)\bigr)
   + \gamma(n) \sum_{s'} p\bigl(s' \mid s,(\mathit{TP} \cdot \pi)_{n-1}(s)\bigr)
        V_{n}^{\pi}(s').
\end{align*}
From Step 2 we already know
\[
Q_{n}^{\pi}\bigl(s', \pi_{n}(s')\bigr)
\le Q_{n}^{\pi^{(n)}}\bigl(s', \pi^{(n)}_{n}(s')\bigr)
\quad \forall\, s' \in \mathcal{S},
\]
which implies $V_{n}^{\pi}(s') \le V_{n}^{\pi^{(n)}}(s')$ for all $s'$. Using
this in the expression above yields
\begin{align*}
Q_{n-1}^{\pi}\bigl(s, \pi_{n-1}(s)\bigr)
&\le r_{n-1}\bigl(s, (\mathit{TP} \cdot \pi)_{n-1}(s)\bigr)
   + \gamma(n) \sum_{s'} p\bigl(s' \mid s,(\mathit{TP} \cdot \pi)_{n-1}(s)\bigr)
        V_{n}^{\pi^{(n)}}(s') \\
&= Q_{n-1}^{\pi^{(n)}}\bigl(s, \pi^{(n)}_{n-1}(s)\bigr).
\end{align*}
Thus
\[
Q_{n-1}^{\pi}\bigl(s, \pi_{n-1}(s)\bigr)
\;\le\; Q_{n-1}^{\pi^{(n)}}\bigl(s, \pi^{(n)}_{n-1}(s)\bigr)
\quad \forall\, s \in \mathcal{S}.
\]

Repeating this argument inductively for $t = n-2, n-3, \dots, 0$, we obtain
\[
Q_{t}^{\pi}\bigl(s, \pi_{t}(s)\bigr)
\;\le\; Q_{t}^{\pi^{(n)}}\bigl(s, \pi^{(n)}_{t}(s)\bigr)
\quad \forall\, s \in \mathcal{S},\ \forall\, t \le n.
\]

\medskip
\noindent\textbf{Step 5: Value comparison for all $t$.}
Combining the three cases $t = n$, $t > n$, and $t < n$, we have
\[
Q_{t}^{\pi}\bigl(s, \pi_{t}(s)\bigr)
\;\le\; Q_{t}^{\pi^{(n)}}\bigl(s, \pi^{(n)}_{t}(s)\bigr)
\quad \forall\, s \in \mathcal{S},\ \forall\, t \in \mathbb{N}.
\]
Equivalently,
\[
V_{t}^{\pi}(s)
= Q_{t}^{\pi}\bigl(s, \pi_{t}(s)\bigr)
\;\le\; Q_{t}^{\pi^{(n)}}\bigl(s, \pi^{(n)}_{t}(s)\bigr)
= V_{t}^{\pi^{(n)}}(s),
\quad \forall\, s \in \mathcal{S},\ \forall\, t \in \mathbb{N}.
\]

\medskip
\noindent\textbf{Step 6: Passage to the fully improved policy.}
In the finite-horizon setting, the discount factor satisfies $\gamma_{T}(t) = 0$
for $t > T$, so $Q_{t}^{\pi'} \equiv 0$ and $V_{t}^{\pi'} \equiv 0$ for any
policy $\pi'$ and all $t > T$. In particular,
\[
V_{t}^{\pi^{(\infty)}}(s) = V_{t}^{\pi^{(T)}}(s)
\quad \forall\, t > T,\ \forall\, s \in \mathcal{S}.
\]
By Lemma~2, for any $t \le T$ we can express
\begin{align*}
V_{t}^{\pi^{(\infty)}}(s)
&= \sum_{s'} C\bigl(\pi^{(\infty)}_{t:T}, s'\bigr)
        V_{T+1}^{\pi^{(\infty)}}(s')
   + C\bigl(\pi^{(\infty)}_{t:T}\bigr) \\
&= \sum_{s'} C\bigl(\pi^{(T)}_{t:T}, s'\bigr)
        V_{T+1}^{\pi^{(T)}}(s')
   + C\bigl(\pi^{(T)}_{t:T}\bigr),
\end{align*}
since by construction $\pi^{(\infty)}_{t:T} = \pi^{(T)}_{t:T}$. Thus
\[
V_{t}^{\pi^{(\infty)}}(s) = V_{t}^{\pi^{(T)}}(s)
\quad \forall\, t \in \mathbb{N},\ \forall\, s \in \mathcal{S}.
\]

Putting everything together, for all $t$ and $s$,
\[
V_{t}^{\pi}(s)
\;\le\; V_{t}^{\pi^{(n)}}(s)
\quad \forall\, n,
\]
and in particular,
\[
V_{t}^{\pi}(s)
\;\le\; V_{t}^{\pi^{(\infty)}}(s)
= V_{t}^{\mathit{TP} \cdot \pi}(s),
\quad \forall\, s \in \mathcal{S},\ \forall\, t \in \mathbb{N}.
\]

Since $\pi^{(\infty)} = \mathit{TP} \cdot \pi$ by definition, this establishes
\[
V^{\pi}_{t}(s) \;\le\; V^{\mathit{TP} \cdot \pi}_{t}(s)
\quad \forall\, s \in \mathcal{S},\ \forall\, t \in \mathbb{N},
\]
which completes the proof.
\end{proof}

\begin{definition}[Greedy policy w.r.t.\ a value function]\label{def:greedy-policy}
Given a value function $V \in \mathcal{V}$, the \emph{greedy policy} 
$\pi^{V} \in \Pi$ is defined, for all $s \in \mathcal{S}$ and $t \in \mathbb{N}$, by
\[
\pi^{V}_{t}(s)
:= \arg\max_{a \in \mathcal{A}}
\Bigl[
    r_{t}(s,a)
    + \gamma(t+1) \sum_{s'} p(s' \mid s,a)\, V_{t+1}(s')
\Bigr].
\]
\end{definition}

\begin{theorem}[Existence of an optimal time-varying policy]\label{thm:optimal-policy}
There exists an optimal time-varying policy $\pi^{*} \in \Pi$ such that
\[
V_{t}^{\pi^{*}}(s) \;\ge\; V_{t}^{\pi}(s)
\quad \forall\, s \in \mathcal{S},\ \forall\, t \in \mathbb{N},\ \forall\, \pi \in \Pi.
\]
\end{theorem}

\begin{proof}
By Theorem~\ref{thm:optimal-V}, there exists an optimal value function
$V^{*} \in \mathcal{V}$ satisfying the Bellman optimality equation
\[
V^{*}_{t}(s)
= \max_{a}
\Bigl[
    r_{t}(s,a)
    + \gamma(t+1) \sum_{s'} p(s' \mid s,a)\, V^{*}_{t+1}(s')
\Bigr].
\]
Define the greedy policy $\pi^{*} := \pi^{V^{*}}$ as in
Definition~\ref{def:greedy-policy}:
\[
\pi^{*}_{t}(s)
\in \arg\max_{a}
\Bigl[
    r_{t}(s,a)
    + \gamma(t+1) \sum_{s'} p(s' \mid s,a)\, V^{*}_{t+1}(s')
\Bigr].
\]

By construction, for all $(s,t)$ we then have
\[
V^{*}_{t}(s)
= r_{t}\bigl(s,\pi^{*}_{t}(s)\bigr)
  + \gamma(t+1) \sum_{s'} p\bigl(s' \mid s,\pi^{*}_{t}(s)\bigr)\,
    V^{*}_{t+1}(s').
\]
This is exactly the Bellman equation for the value function of policy $\pi^{*}$.
By uniqueness of solutions to the Bellman equation for a fixed policy in this
finite-horizon setting, it follows that
\[
V^{\pi^{*}}_{t}(s) = V^{*}_{t}(s)
\quad \forall\, s \in \mathcal{S},\ \forall\, t \in \mathbb{N}.
\]

Since $V^{*}$ is optimal in $\mathcal{V}$, we have
\[
V^{\pi}_{t}(s) \;\le\; V^{*}_{t}(s)
= V^{\pi^{*}}_{t}(s)
\quad \forall\, s \in \mathcal{S},\ \forall\, t \in \mathbb{N},\ \forall\, \pi \in \Pi.
\]
Thus $\pi^{*}$ is an optimal time-varying policy.
\end{proof}

\begin{theorem}[Convergence of policy iteration]\label{thm:policy-iteration}
Starting from any policy $\pi \in \Pi$, recursively applying the Bellman policy
operator $\mathit{TP}$ converges to the optimal policy. In particular,
\[
\lim_{n \to \infty} V^{\mathit{TP}^{n} \cdot \pi}_{t}(s)
= V^{\pi^{*}}_{t}(s)
\quad \forall\, (s,t) \in \mathcal{S} \times \mathbb{N},
\]
where $\pi^{*}$ is an optimal time-varying policy as in
Theorem~\ref{thm:optimal-policy}.
\end{theorem}

\begin{proof}
Fix an arbitrary initial policy $\pi \in \Pi$. Consider the value function
$V^{\mathit{TP} \cdot \pi}$ of the one-step improved policy $\mathit{TP} \cdot \pi$.
By the Bellman equation for $V^{\mathit{TP} \cdot \pi}$, for any $(s,t)$ we have
\begin{align*}
V^{\mathit{TP} \cdot \pi}_{t}(s)
&= r_{t}\bigl(s, (\mathit{TP}\cdot \pi)_{t}(s)\bigr)
  + \gamma(t+1) \sum_{s'} p\bigl(s' \mid s, (\mathit{TP}\cdot \pi)_{t}(s)\bigr)
        V^{\mathit{TP} \cdot \pi}_{t+1}(s') \\
&= r_{t}\bigl(s, (\mathit{TP}\cdot \pi)_{t}(s)\bigr)
  + \gamma(t+1) \sum_{s'} p\bigl(s' \mid s, (\mathit{TP}\cdot \pi)_{t}(s)\bigr)
        \Bigl( V^{\mathit{TP} \cdot \pi}_{t+1}(s') - V^{\pi}_{t+1}(s') + V^{\pi}_{t+1}(s') \Bigr) \\
&= \underbrace{\gamma(t+1) \sum_{s'} p\bigl(s' \mid s, (\mathit{TP}\cdot \pi)_{t}(s)\bigr)
        \bigl( V^{\mathit{TP} \cdot \pi}_{t+1}(s') - V^{\pi}_{t+1}(s') \bigr)}_{\text{(I)}} \\
&\quad\quad
 + \Bigl[
        r_{t}\bigl(s, (\mathit{TP}\cdot \pi)_{t}(s)\bigr)
        + \gamma(t+1) \sum_{s'} p\bigl(s' \mid s, (\mathit{TP}\cdot \pi)_{t}(s)\bigr)
            V^{\pi}_{t+1}(s')
   \Bigr] \\
&= \text{(I)} + \max_{a}
    \Bigl[
        r_{t}(s,a)
        + \gamma(t+1) \sum_{s'} p(s' \mid s,a)\, V^{\pi}_{t+1}(s')
    \Bigr] \\
&= \text{(I)} + (TV^{\pi})_{t}(s),
\end{align*}
where the last two equalities use the definition of the greedy action
$(\mathit{TP} \cdot \pi)_{t}(s)$ and of the Bellman value operator $T$.

By Theorem~\ref{thm:policy-improvement} we know that
\[
V^{\mathit{TP} \cdot \pi}_{t+1}(s') \;\ge\; V^{\pi}_{t+1}(s')
\quad \forall\, s' \in \mathcal{S},\ \forall\, t,
\]
so every term in
\[
V^{\mathit{TP} \cdot \pi}_{t+1}(s') - V^{\pi}_{t+1}(s')
\]
is nonnegative. Since $p(\cdot \mid s,(\mathit{TP}\cdot \pi)_{t}(s))$ is a
probability distribution and $\gamma(t+1) \ge 0$, we have
\[
\text{(I)} \;\ge\; 0.
\]
Thus
\[
V^{\mathit{TP} \cdot \pi}_{t}(s)
\;\ge\; (TV^{\pi})_{t}(s)
\quad \forall\, s \in \mathcal{S},\ \forall\, t \in \mathbb{N}.
\]

Iterating this inequality, we obtain for any $n \ge 1$,
\[
V^{\mathit{TP}^{n} \cdot \pi}_{t}(s)
\;\ge\; (T^{n}V^{\pi})_{t}(s)
\quad \forall\, s \in \mathcal{S},\ \forall\, t \in \mathbb{N}.
\]
Taking limits as $n \to \infty$ and using Theorem~\ref{thm:optimal-V}, which
states that $T^{n}V^{\pi} \to V^{*}$ (the unique optimal value function), we get
\[
\lim_{n \to \infty} V^{\mathit{TP}^{n} \cdot \pi}_{t}(s)
\;\ge\; \lim_{n \to \infty} (T^{n}V^{\pi})_{t}(s)
= V^{*}_{t}(s)
= V^{\pi^{*}}_{t}(s),
\]
where the last equality uses Theorem~\ref{thm:optimal-policy}:
$V^{\pi^{*}} = V^{*}$.

On the other hand, for every $n$,
\[
V^{\mathit{TP}^{n} \cdot \pi}_{t}(s)
\;\le\; V^{\pi^{*}}_{t}(s)
\quad \forall\, s,t,
\]
because $\pi^{*}$ is optimal and thus dominates every policy, including
$\mathit{TP}^{n} \cdot \pi$. Passing to the limit yields
\[
\lim_{n \to \infty} V^{\mathit{TP}^{n} \cdot \pi}_{t}(s)
\;\le\; V^{\pi^{*}}_{t}(s).
\]

Combining both inequalities, we conclude
\[
\lim_{n \to \infty} V^{\mathit{TP}^{n} \cdot \pi}_{t}(s)
= V^{\pi^{*}}_{t}(s)
\quad \forall\, (s,t) \in \mathcal{S} \times \mathbb{N},
\]
which proves the theorem.
\end{proof}

\begin{theorem}\label{thm:static-gap}
   There exists a finite-horizon dynamic MDP (DMDP) for which an
optimal time-varying policy achieves strictly higher value than any static
(time-invariant) policy.
\end{theorem}

\begin{proof}
We construct a finite-horizon DMDP with:
\[
\mathcal{S} = \{s\}, \quad \mathcal{A} = \{a_1, a_2\}, \quad T = 1,\quad
P(s \mid s, a) = 1 \ \forall a,\quad \gamma(t) \equiv 1.
\]
The reward is time-varying and given by
\[
r_0(s, a_1) = 0,\quad r_0(s, a_2) = 1, \qquad
r_1(s, a_1) = 1,\quad r_1(s, a_2) = 0.
\]

A \emph{static} policy is one that does not depend on $t$, i.e.
$\pi_t(s) \equiv \bar{\pi}(s)$ for all $t$. Since there is only one state,
any static policy must either always choose $a_1$ or always choose $a_2$:
\[
\bar{\pi}(s) = a_1
\quad \text{or} \quad
\bar{\pi}(s) = a_2.
\]

Compute the value at $t=0$ for each static policy:
\begin{itemize}
    \item If $\bar{\pi}(s) = a_1$, then
    \[
    V_0^{\bar{\pi}}(s)
    = r_0(s,a_1) + r_1(s,a_1)
    = 0 + 1 = 1.
    \]
    \item If $\bar{\pi}(s) = a_2$, then
    \[
    V_0^{\bar{\pi}}(s)
    = r_0(s,a_2) + r_1(s,a_2)
    = 1 + 0 = 1.
    \]
\end{itemize}
So any static policy attains value $V_0^{\bar{\pi}}(s) = 1$.

Now consider the time-varying policy $\tilde{\pi}$ defined by
\[
\tilde{\pi}_0(s) = a_2, \qquad \tilde{\pi}_1(s) = a_1.
\]
Its value at $t=0$ is
\[
V_0^{\tilde{\pi}}(s)
= r_0(s, a_2) + r_1(s, a_1)
= 1 + 1 = 2.
\]

Thus,
\[
\sup_{\pi \ \text{static}} V_0^{\pi}(s) = 1
\quad\text{but}\quad
\sup_{\pi \ \text{time-varying}} V_0^{\pi}(s) \ge V_0^{\tilde{\pi}}(s) = 2.
\]
Hence allowing policies to depend on time strictly improves the optimal value
in this DMDP. This proves the claim.
\end{proof}


\begin{theorem}[Value of a concatenated policy]\label{thm:concat-value}
Under Assumptions~\hyperref[asm:A1]{A1}--\hyperref[asm:A3]{A3},
let $\pi = \pi^{1}_{0:T_{1}} \circ \pi^{2}_{0:T_{2}}$ be as in
Definition~\ref{def:concat-policy} and let $T := T_{1} + T_{2}$. Then for all
$t = 0,\dots,T-1$ and $s \in \mathcal{S}$,
\begin{equation}\label{eq:concat-compact}
V_{t}^{\pi}(s)
=
V^{\pi^{1}}_{t}(s)
+ \gamma^{\max(T_{1}-t,0)}
  \,\mathbb{E}\!\Big[
      V^{\pi^{2}}_{\max(0,\,t-T_{1})}\bigl(s_{\max(t,T_{1})}\bigr)
      \,\Big|\, s_{t} = s
    \Big].
\end{equation}
Equivalently, for $t < T$,
\[
V_{t}^{\pi}(s)
=
\begin{cases}
V_{t}^{\pi^{1}}(s)
+ \gamma^{T_{1}-t}\,
  \mathbb{E}\!\big[
    V_{0}^{\pi^{2}}(s_{T_{1}}) \mid s_{t} = s
  \big],
& t < T_{1},\\[4pt]
V_{t-T_{1}}^{\pi^{2}}(s),
& T_{1} \le t < T.
\end{cases}
\]
\end{theorem}

\begin{proof}
By definition of the finite-horizon value function for a time-varying policy,
we split the discounted return at time $T_{1}$. For $t < T_{1}$, the first
$T_{1}-t$ rewards are collected under $\pi^{1}$ and the remaining $T_{2}$
rewards under $\pi^{2}$ starting from $s_{T_{1}}$. Taking expectations over
$s_{T_{1}}\sim\pi^{1}$ and applying the tower property yields the stated
decomposition. For $T_{1} \le t < T$, the policy is entirely $\pi^{2}$ and the
value reduces to $V_{t-T_{1}}^{\pi^{2}}(s)$.
\end{proof}


\begin{theorem}[GDS optimality for optimal reach]\label{thm:gds-reach}
Under Assumptions~\hyperref[asm:A1]{A1}--\hyperref[asm:A3]{A3}, for any reachable goal set, General Dijkstra Search for Optimal Reach finds an
optimal goal-reaching policy. Specifically, for every
$\mathcal{G} \in \mathcal{G}_{T}^{\supset}(s)$, there exists
$\pi^{*} \in \Pi_{1:T}^{\mathcal{G}\supset}(s)$ such that for every
$\pi \in \Pi_{1:T}^{\mathcal{G}\supset}(s)$,
\[
V_{0}^{\pi^{*}}(s) \ge V_{0}^{\pi}(s).
\]
\end{theorem}

\begin{proof}
The proof follows the same three-step strategy as the goal-covering analog in Appendix~\ref{app:gds-coverage}: (i) the queue invariant shows that every queued element $(\pi_{1:t},v_t,\mathcal{G}^{\pi_{1:t}}(s),t)$ satisfies $v_t = V_0^{\pi_{1:t}}(s)$; (ii) any popped policy has value at least as large as every queued or future-queued policy, using Theorem~\ref{thm:concat-value} and $r_t\le 0$ to bound continuation values; (iii) the pruning step discards only policies dominated by a strictly better one, so no optimal policy is lost. Combining these, the first popped policy whose goal set is contained in the target $\mathcal{G}^{*}$ is optimal.
\end{proof}


% Optionally include supplemental material (complete proofs, additional experiments and plots) in appendix.
% All such materials \textbf{SHOULD be included in the main submission.}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% \newpage


\section{Symmetric goal-covering result}
\label{app:gds-coverage}

For completeness we record the dual of the goal-reaching algorithm and theorem
of \S\ref{sec:gds}. The covering case swaps the direction of the goal-set
inclusion: a policy is \emph{covering} a target $\mathcal{G}$ if its reachable
goal set contains $\mathcal{G}$, rather than being contained in it.

\begin{definition}[Covering policies]
Given a start state $s \in \mathcal{S}$ and a set of goal states
$\mathcal{G} \subset \mathcal{S}$, the set of \emph{covering policies}
$\Pi^{\mathcal{G}\subset}_{1:T}(s)$ is
\[
\Pi^{\mathcal{G}\subset}_{1:T}(s)
:= \Bigl\{ \pi \in \bigcup_{t=1}^{T} \Pi_{t} \,\Big|\, \mathcal{G} \subset \mathcal{G}^{\pi}(s) \Bigr\}.
\]
\end{definition}

\begin{definition}[Optimal goal-covering policy]
An \emph{optimal goal-covering policy} is a policy
$\pi^{*} \in \Pi^{\mathcal{G}\subset}_{1:T}(s)$ such that, for all
$\pi \in \Pi^{\mathcal{G}\subset}_{1:T}(s)$,
$V_{0}^{\pi^{*}}(s) \geq V_{0}^{\pi}(s)$.
\end{definition}

\begin{algorithm}[H]
\caption{General Dijkstra Search (Optimal Coverage)}
\KwIn{Error tolerance $\epsilon_{t}:= \frac{r_{\max}}{1-\gamma} \cdot \sum_{i=t}^{\infty} \gamma^{i}$, start state $s \in \mathcal{S}$, goal states $\mathcal{G}^{*} \subset \mathcal{S}$}
Initialize $\mathcal{Q} = \{(\emptyset, 0, \{s\}, 0)\}$, $v = \emptyset$, $\mathcal{R} = \emptyset$\;
\While{$\mathcal{Q} \neq \emptyset$}{
Pop $(\pi_{1:t}, v_{t}, \mathcal{G}^{\pi_{1:t}}(s), t)$ from the priority queue with maximal value; add $\mathcal{G}^{\pi_{1:t}}(s)$ into $\mathcal{R}$ if it is not already inside\;
\If{$\mathcal{G}^{*} \subset \mathcal{G}^{\pi_{1:t}}(s)$}{
break\;
}
\If{$(\exists (s, \mathcal{G}) \in v \text{ s.t. } \mathcal{G}^{\pi_{1:t}}(s) \subset \mathcal{G} \text{ and } v_{t} \leq v(s,\mathcal{G}) - \epsilon_{t})$ or $t = T$}{
continue\;
}
\For{$\pi \in \Pi_{1}$}{
Concatenate policy $\pi_{1:t+1} = \pi_{1:t} \circ \pi$ and compute $v_{t+1} = v_{t} + \gamma^{t} \cdot \mathbb{E}_{s_{0:t}}\big[ V_{0}^{\pi}(s_{t}) \,\big|\, s_{0} = s\big]$\;
Push $(\pi_{1:t+1}, v_{t+1}, \mathcal{G}^{\pi_{1:t+1}}(s), t+1)$ into $\mathcal{Q}$\;
\ForEach{$(s, \mathcal{G}) \in v$ such that $\mathcal{G} \subset \mathcal{G}^{\pi_{1:t+1}}(s)$ and $\mathcal{G} \notin \mathcal{R}$}{
$v(s, \mathcal{G}) \gets \max(v(s, \mathcal{G}), v_{t+1})$\;
}
\If{$(s, \mathcal{G}^{\pi_{1:t}}) \notin v$}{
$v(s, \mathcal{G}^{\pi_{1:t}}) \gets v_{t+1}$\;
}
}
}
\end{algorithm}

\begin{theorem}\label{thm:gds}
Under Assumptions~\hyperref[asm:A1]{A1}--\hyperref[asm:A3]{A3}, for any coverable goal set, General Dijkstra Search
for Optimal Coverage finds an optimal goal-covering policy. Specifically,
for every $\mathcal{G} \in \mathcal{G}_{T}^{\subset}(s)$, there exists
$\pi^{*} \in \Pi_{1:T}^{\mathcal{G}\subset}(s)$ such that for every
$\pi \in \Pi_{1:T}^{\mathcal{G}\subset}(s)$,
$V_{0}^{\pi^{*}}(s) \ge V_{0}^{\pi}(s)$.
\end{theorem}

\begin{proof}
By the analogous coverage-side queue invariant, the coverage pruning guarantee,
and the fact that the first popped feasible covering policy is optimal among
all policies covering the same goal set. The argument mirrors the reaching case
(\S\ref{sec:gds}) verbatim with the inclusion direction flipped.
\end{proof}

\section{Policy Composition}

\begin{definition}[Policy dominance]
Let $\pi^{1}, \pi^{2} \in \bigcup_{t=1}^{T} \Pi_{t}$. We say that
$\pi^{1}$ is dominated by $\pi^{2}$ if
\[
\mathcal{G}^{\pi^{1}}(s) \subset \mathcal{G}^{\pi^{2}}(s).
\]
\end{definition}

\begin{definition}[Dominating and dominated goal sets]
Let $\mathcal{G} \subset \mathcal{S}$ and let
$\pi \in \bigcup_{t=1}^{T} \Pi_{t}$.
\begin{enumerate}
    \item A policy $\pi$ is said to \emph{dominate} a goal set
    $\mathcal{G}$ when starting from $s \in \mathcal{S}$ if
    \[
    \mathcal{G} \subset \mathcal{G}^{\pi}(s).
    \]
    \item A policy $\pi$ is said to be \emph{dominated by} a goal set
    $\mathcal{G}$ when starting from $s \in \mathcal{S}$ if
    \[
    \mathcal{G}^{\pi}(s) \subset \mathcal{G}.
    \]
\end{enumerate}
\end{definition}

\begin{lemma}
If $\pi^{1,*}$ dominates $\pi^{1}$, then
$\pi^{1,*} \circ \pi^{2}$ also dominates $\pi^{1} \circ \pi^{2}$.
\end{lemma}

\begin{proof}
Since $\pi^{1,*}$ dominates $\pi^{1}$, we have
\[
\mathcal{G}^{\pi^{1}}(s) \subset \mathcal{G}^{\pi^{1,*}}(s).
\]
Therefore,
\[
\mathcal{G}^{\pi^{1} \circ \pi^{2}}(s)
= \bigcup_{s' \in \mathcal{G}^{\pi^{1}}(s)} \mathcal{G}^{\pi^{2}}(s')
\subset
\bigcup_{s' \in \mathcal{G}^{\pi^{1,*}}(s)} \mathcal{G}^{\pi^{2}}(s')
= \mathcal{G}^{\pi^{1,*} \circ \pi^{2}}(s).
\]
Hence $\pi^{1,*} \circ \pi^{2}$ dominates $\pi^{1} \circ \pi^{2}$.
\end{proof}

\begin{lemma}
In the General Dijkstra Search algorithms for optimal reach and optimal
coverage, any element $(\pi_{1:t}, v_{t}, \mathcal{G}^{\pi_{1:t}}(s), t)$
within the priority queue satisfies
\[
v_{t} = V_{0}^{\pi_{1:t}}(s).
\]
\end{lemma}

\begin{proof}
We prove the claim by induction on the horizon length $t$ of a policy in the
queue.

For the base case $t=1$, any element
$(\pi_{1}, v_{1}, \mathcal{G}^{\pi_{1}}(s), 1) \in \mathcal{Q}$ with
$\pi_{1} \in \Pi_{1}$ is constructed by concatenating a one-step policy
$\pi \in \Pi_{1}$ with the empty policy inserted at initialization. Hence
$\pi_{1} = \pi$ and $v_{0}=0$, so
\[
\begin{aligned}
v_{1}
&= 0 + \gamma^{0} \cdot \mathbb{E}_{s_{0}}\Big[ V_{0}^{\pi}(s_{0}) \,\big|\, s_{0}=s\Big] \\
&= V_{0}^{\pi_{1}}(s).
\end{aligned}
\]

Now assume that for some $n \le T$, every element
$(\pi_{1:t}, v_{t}, \mathcal{G}^{\pi_{1:t}}(s), t) \in \mathcal{Q}$ with
$0 \le t \le n-1$ satisfies
\[
v_{t} = V_{0}^{\pi_{1:t}}(s).
\]

Consider any element
$(\pi_{1:t+1}, v_{t+1}, \mathcal{G}^{\pi_{1:t+1}}(s), t+1) \in \mathcal{Q}$.
By construction, it is obtained by extending some
$(\pi_{1:t}, v_{t}, \mathcal{G}^{\pi_{1:t}}(s), t) \in \mathcal{Q}$ by a
one-step policy $\pi_{t+1} \in \Pi_{1}$. Therefore,
\[
\begin{aligned}
v_{t+1}
&= v_{t} + \gamma^{t} \cdot \mathbb{E}_{s_{0:t}}\Big[ V_{0}^{\pi_{t+1}}(s_{t}) \,\big|\, s_{0} = s\Big] \\
&= V_{0}^{\pi_{1:t}}(s) + \gamma^{t} \cdot \mathbb{E}_{s_{0:t}}\Big[ V_{0}^{\pi_{t+1}}(s_{t}) \,\big|\, s_{0} = s\Big] \\
&= V_{0}^{\pi_{1:t+1}}(s),
\end{aligned}
\]
where the second equality uses the induction hypothesis and the third equality
follows from Theorem~\ref{thm:concat-value}. This completes the induction.
\end{proof}

\begin{lemma}
Any popped policy $\pi^{*}$ within the General Dijkstra Search algorithms for
optimal reach and optimal coverage has larger value at $t=0$ than any policy
$\pi$ that is either currently in the queue or will be added to the queue.
Specifically,
\[
V_{0}^{\pi^{*}}(s) \ge V_{0}^{\pi}(s).
\]
\end{lemma}

\begin{proof}
Let $\pi^{*}$ be the policy popped at some step of the algorithm. Every policy
in $\bigcup_{t=1}^{T} \Pi_{t}$ belongs to one of the following groups:
\begin{enumerate}
    \item policies already popped from $\mathcal{Q}$,
    \item policies currently in $\mathcal{Q}$,
    \item policies that will be added to $\mathcal{Q}$ in the future,
    \item policies that never appear in $\mathcal{Q}$.
\end{enumerate}

Since $\pi^{*}$ is popped from the priority queue, it has value at least as
large as every policy currently in the queue. Hence for every policy
$\pi$ in group (2),
\[
V_{0}^{\pi^{*}}(s) \ge V_{0}^{\pi}(s).
\]

Now let $\pi$ be a policy that will be added to the queue in the future. Then
$\pi$ must extend some current queue element $\pi_{1:t_{1}}$ with $t_{1} < T$.
By Theorem~\ref{thm:concat-value},
\[
V_{0}^{\pi}(s)
= V_{0}^{\pi_{1:t_{1}}}(s)
  + \gamma^{t_{1}} \cdot
    \mathbb{E}_{s_{0:t_{1}} \sim \pi_{1:t_{1}}}
    \Big[ V_{0}^{\pi_{t_{1}+1:t}}(s_{t_{1}}) \,\big|\, s_{0}=s\Big].
\]
By Lemma~3, the continuation value satisfies
$V_{0}^{\pi_{t_{1}+1:t}}(s') \le 0$ for all $s' \in \mathcal{S}$. Therefore,
\[
V_{0}^{\pi}(s) \le V_{0}^{\pi_{1:t_{1}}}(s).
\]
Since $\pi_{1:t_{1}}$ is currently in the queue or is equal to $\pi^{*}$, the
previous argument implies
\[
V_{0}^{\pi}(s) \le V_{0}^{\pi_{1:t_{1}}}(s) \le V_{0}^{\pi^{*}}(s).
\]
This proves the claim for all policies currently in the queue or to be added
to the queue.
\end{proof}

\begin{lemma}
Under Assumption~4, in General Dijkstra Search (Optimal Reach), any policy
$\pi \in \Pi_{t}$ with $t \le T$ that never appears in the queue must
dominate a popped and unskipped policy $\pi^{*}$ such that
\[
V_{0}^{\pi}(s) + \epsilon_{t} \le V_{0}^{\pi^{*}}(s)
\qquad \text{and} \qquad
\mathcal{G}^{\pi}(s) \supset \mathcal{G}^{\pi^{*}}(s).
\]
\end{lemma}

\begin{proof}
Let $\pi \in \Pi_{t}$ with $t \le T$ be a policy that never appears in
$\mathcal{Q}$. Then there exists an index $1 \le t' < t$ such that
$\pi_{1:t'}$ is the longest prefix of $\pi$ that is popped and skipped by the
algorithm. We prove by induction on
$n \in \{t', t'+1, \dots, t\}$ that there exists a popped and unskipped policy
$\pi_{n}^{*}$ dominated by $\pi_{1:n}$ and satisfying
\[
V_{0}^{\pi_{n}^{*}}(s) - V_{0}^{\pi_{1:n}}(s) \ge \epsilon_{n}.
\]

For the base case $n=t'$, since $\pi_{1:t'}$ is popped and skipped, by the
pruning rule in Algorithm~1 there exists a popped and unskipped policy
$\pi_{t'}^{*}$ such that
\[
\mathcal{G}^{\pi_{1:t'}}(s) \supset \mathcal{G}^{\pi_{t'}^{*}}(s)
\qquad \text{and} \qquad
V_{0}^{\pi_{t'}^{*}}(s) - V_{0}^{\pi_{1:t'}}(s) \ge \epsilon_{t'}.
\]

Now assume that for some $t_{0} \in \{t', t'+1, \dots, t-1\}$ there exists a
popped and unskipped policy $\pi_{t_{0}}^{*}$ dominated by $\pi_{1:t_{0}}$ such
that
\[
V_{0}^{\pi_{t_{0}}^{*}}(s) - V_{0}^{\pi_{1:t_{0}}}(s) \ge \epsilon_{t_{0}}.
\]
Because $\pi_{t_{0}}^{*}$ is popped and unskipped, all of its one-step
extensions are added to the queue, including
$\pi_{t_{0}}^{*} \circ \pi_{t_{0}+1}$. We consider two cases.

\medskip
\noindent\textbf{Case 1.}
$\pi_{t_{0}}^{*} \circ \pi_{t_{0}+1}$ is popped and unskipped. Then
\[
\begin{aligned}
&V_{0}^{\pi_{t_{0}}^{*} \circ \pi_{t_{0}+1}}(s)
 - V_{0}^{\pi_{1:t_{0}+1}}(s) \\
&= \Bigl(V_{0}^{\pi_{t_{0}}^{*}}(s) - V_{0}^{\pi_{1:t_{0}}}(s)\Bigr)
 + \gamma^{t_{0}} \Bigl(
    \mathbb{E}_{s_{0:t_{0}} \sim \pi_{t_{0}}^{*}}
    \bigl[V_{0}^{\pi_{t_{0}+1}}(s_{t_{0}}) \mid s_{0}=s\bigr]
    -
    \mathbb{E}_{s_{0:t_{0}} \sim \pi_{1:t_{0}}}
    \bigl[V_{0}^{\pi_{t_{0}+1}}(s_{t_{0}}) \mid s_{0}=s\bigr]
   \Bigr).
\end{aligned}
\]
By the induction hypothesis, the first term is at least $\epsilon_{t_{0}}$.
By Lemma~3, each continuation value is bounded below by
$-\frac{r_{\max}}{1-\gamma}$ and above by $0$, hence the difference in
parentheses is bounded below by $-\frac{r_{\max}}{1-\gamma}$. Therefore,
\[
V_{0}^{\pi_{t_{0}}^{*} \circ \pi_{t_{0}+1}}(s)
 - V_{0}^{\pi_{1:t_{0}+1}}(s)
\ge
\epsilon_{t_{0}} - \gamma^{t_{0}} \frac{r_{\max}}{1-\gamma}
= \epsilon_{t_{0}+1}.
\]
By Lemma~4,
$\pi_{t_{0}}^{*} \circ \pi_{t_{0}+1}$ is dominated by $\pi_{1:t_{0}+1}$.

\medskip
\noindent\textbf{Case 2.}
$\pi_{t_{0}}^{*} \circ \pi_{t_{0}+1}$ is popped and skipped. Then by the
pruning rule in Algorithm~1 there exists a popped and unskipped policy
$\pi_{t_{0}+1}^{*}$ dominated by
$\pi_{t_{0}}^{*} \circ \pi_{t_{0}+1}$ such that
\[
V_{0}^{\pi_{t_{0}+1}^{*}}(s)
 - V_{0}^{\pi_{t_{0}}^{*} \circ \pi_{t_{0}+1}}(s)
\ge \epsilon_{t_{0}+1}.
\]
Since domination is transitive and Lemma~4 implies that
$\pi_{t_{0}}^{*} \circ \pi_{t_{0}+1}$ is dominated by $\pi_{1:t_{0}+1}$, the
policy $\pi_{t_{0}+1}^{*}$ is also dominated by $\pi_{1:t_{0}+1}$. Moreover,
combining the previous display with the lower bound from Case 1 gives
\[
\begin{aligned}
V_{0}^{\pi_{t_{0}+1}^{*}}(s) - V_{0}^{\pi_{1:t_{0}+1}}(s)
&=
\Bigl(V_{0}^{\pi_{t_{0}+1}^{*}}(s)
 - V_{0}^{\pi_{t_{0}}^{*} \circ \pi_{t_{0}+1}}(s)\Bigr)
 +
\Bigl(V_{0}^{\pi_{t_{0}}^{*} \circ \pi_{t_{0}+1}}(s)
 - V_{0}^{\pi_{1:t_{0}+1}}(s)\Bigr) \\
&\ge \epsilon_{t_{0}+1}.
\end{aligned}
\]

In either case, we obtain a popped and unskipped policy dominated by
$\pi_{1:t_{0}+1}$ and satisfying the desired value gap. By induction, this
holds up to $n=t$. Hence there exists a popped and unskipped policy
$\pi^{*}$ dominated by $\pi$ such that
\[
V_{0}^{\pi^{*}}(s) - V_{0}^{\pi}(s) \ge \epsilon_{t}.
\]
Equivalently,
\[
V_{0}^{\pi}(s) + \epsilon_{t} \le V_{0}^{\pi^{*}}(s)
\qquad \text{and} \qquad
\mathcal{G}^{\pi}(s) \supset \mathcal{G}^{\pi^{*}}(s).
\]
This proves the claim.
\end{proof}

\begin{lemma}
In General Dijkstra Search for Optimal Reach (Algorithm~1) starting from
$s \in \mathcal{S}$, let $\pi^{*}$ be a popped and unskipped policy that does
not dominate any previous policy in the queue. Then $\pi^{*}$ is optimal among
all policies that it dominates. Equivalently, if
$\mathcal{G} = \mathcal{G}^{\pi^{*}}(s)$, then for every
$\pi \in \Pi_{1:T}^{\mathcal{G}\supset}(s)$,
\[
V_{0}^{\pi^{*}}(s) \ge V_{0}^{\pi}(s).
\]
\end{lemma}

\begin{proof}
Fix a step $n$ at which $\pi^{*}$ is popped and unskipped. Every policy in
$\bigcup_{t=1}^{T} \Pi_{t}$ belongs to one of the following groups at step $n$:
\begin{enumerate}
    \item policies popped from $\mathcal{Q}$ at some step $\le n$,
    \item policies currently in $\mathcal{Q}$ at step $n$,
    \item policies that will enter $\mathcal{Q}$ at some step $> n$,
    \item policies that never appear in $\mathcal{Q}$.
\end{enumerate}

Because $\pi^{*}$ is the first popped and unskipped policy that does not
dominate any previous policy in the queue, there is no policy in group (1)
whose goal set is contained in $\mathcal{G}^{\pi^{*}}(s)$. That is,
\[
\Bigl\{ \pi \in \bigcup_{t=1}^{T} \Pi_{t}
\,\Big|\, \mathcal{G}^{\pi}(s) \subset \mathcal{G}^{\pi^{*}}(s) \Bigr\}
\cap (1) = \emptyset.
\]

Now let
$\pi \in \bigcup_{t=1}^{T} \Pi_{t}$ satisfy
$\mathcal{G}^{\pi}(s) \subset \mathcal{G}^{\pi^{*}}(s)$. We consider the
possible groups to which $\pi$ belongs.

If $\pi$ belongs to group (2) or group (3), then Lemma~7 gives directly
\[
V_{0}^{\pi^{*}}(s) \ge V_{0}^{\pi}(s).
\]

If $\pi$ belongs to group (4), then by Lemma~8 there exists a popped and
unskipped policy $\pi'$ such that
\[
V_{0}^{\pi'}(s) \ge V_{0}^{\pi}(s)
\qquad \text{and} \qquad
\mathcal{G}^{\pi}(s) \supset \mathcal{G}^{\pi'}(s).
\]
Since $\mathcal{G}^{\pi}(s) \subset \mathcal{G}^{\pi^{*}}(s)$, we also have
\[
\mathcal{G}^{\pi'}(s) \subset \mathcal{G}^{\pi^{*}}(s).
\]
Therefore $\pi'$ cannot belong to group (1), by the defining property of
$\pi^{*}$. It also cannot be $\pi^{*}$-dominating in the queue prior to the pop
of $\pi^{*}$. Hence $\pi'$ must belong to group (3), and Lemma~7 yields
\[
V_{0}^{\pi^{*}}(s) \ge V_{0}^{\pi'}(s).
\]
Combining the two inequalities gives
\[
V_{0}^{\pi^{*}}(s) \ge V_{0}^{\pi'}(s) \ge V_{0}^{\pi}(s).
\]

Thus for every policy $\pi$ with
$\mathcal{G}^{\pi}(s) \subset \mathcal{G}^{\pi^{*}}(s)$, we have
\[
V_{0}^{\pi^{*}}(s) \ge V_{0}^{\pi}(s).
\]
Equivalently, $\pi^{*}$ is the optimal goal-reaching policy for the pair
$(s, \mathcal{G}^{\pi^{*}}(s))$.
\end{proof}

\begin{lemma}
Under Assumption~4, in General Dijkstra Search for Optimal Coverage
(Algorithm~2), any policy $\pi \in \Pi_{t}$ with $t \le T$ that never appears
in the queue is dominated by a popped and unskipped policy $\pi^{*}$ such that
\[
V_{0}^{\pi}(s) + \epsilon_{t} \le V_{0}^{\pi^{*}}(s)
\qquad \text{and} \qquad
\mathcal{G}^{\pi}(s) \subset \mathcal{G}^{\pi^{*}}(s).
\]
\end{lemma}

\begin{proof}
Let $\pi \in \Pi_{t}$ with $t \le T$ be a policy that never appears in
$\mathcal{Q}$. Then there exists an index $1 \le t' < t$ such that
$\pi_{1:t'}$ is the longest prefix of $\pi$ that is popped and skipped by the
algorithm. We prove by induction on
$n \in \{t', t'+1, \dots, t\}$ that there exists a popped and unskipped policy
$\pi_{n}^{*}$ that dominates $\pi_{1:n}$ and satisfies
\[
V_{0}^{\pi_{n}^{*}}(s) - V_{0}^{\pi_{1:n}}(s) \ge \epsilon_{n}.
\]

For the base case $n=t'$, since $\pi_{1:t'}$ is popped and skipped, by the
pruning rule in Algorithm~2 there exists a popped and unskipped policy
$\pi_{t'}^{*}$ such that
\[
\mathcal{G}^{\pi_{1:t'}}(s) \subset \mathcal{G}^{\pi_{t'}^{*}}(s)
\qquad \text{and} \qquad
V_{0}^{\pi_{t'}^{*}}(s) - V_{0}^{\pi_{1:t'}}(s) \ge \epsilon_{t'}.
\]

Now assume that for some $t_{0} \in \{t', t'+1, \dots, t-1\}$ there exists a
popped and unskipped policy $\pi_{t_{0}}^{*}$ that dominates $\pi_{1:t_{0}}$
and satisfies
\[
V_{0}^{\pi_{t_{0}}^{*}}(s) - V_{0}^{\pi_{1:t_{0}}}(s) \ge \epsilon_{t_{0}}.
\]
Because $\pi_{t_{0}}^{*}$ is popped and unskipped, all of its one-step
extensions are added to the queue, including
$\pi_{t_{0}}^{*} \circ \pi_{t_{0}+1}$. We consider two cases.

\medskip
\noindent\textbf{Case 1.}
$\pi_{t_{0}}^{*} \circ \pi_{t_{0}+1}$ is popped and unskipped. Then
\[
\begin{aligned}
&V_{0}^{\pi_{t_{0}}^{*} \circ \pi_{t_{0}+1}}(s)
 - V_{0}^{\pi_{1:t_{0}+1}}(s) \\
&= \Bigl(V_{0}^{\pi_{t_{0}}^{*}}(s) - V_{0}^{\pi_{1:t_{0}}}(s)\Bigr)
 + \gamma^{t_{0}} \Bigl(
    \mathbb{E}_{s_{0:t_{0}} \sim \pi_{t_{0}}^{*}}
    \bigl[V_{0}^{\pi_{t_{0}+1}}(s_{t_{0}}) \mid s_{0}=s\bigr]
    -
    \mathbb{E}_{s_{0:t_{0}} \sim \pi_{1:t_{0}}}
    \bigl[V_{0}^{\pi_{t_{0}+1}}(s_{t_{0}}) \mid s_{0}=s\bigr]
   \Bigr).
\end{aligned}
\]
By the induction hypothesis, the first term is at least $\epsilon_{t_{0}}$.
By Lemma~3, the difference in parentheses is bounded below by
$-\frac{r_{\max}}{1-\gamma}$. Therefore,
\[
V_{0}^{\pi_{t_{0}}^{*} \circ \pi_{t_{0}+1}}(s)
 - V_{0}^{\pi_{1:t_{0}+1}}(s)
\ge
\epsilon_{t_{0}} - \gamma^{t_{0}} \frac{r_{\max}}{1-\gamma}
= \epsilon_{t_{0}+1}.
\]
By Lemma~4,
$\pi_{t_{0}}^{*} \circ \pi_{t_{0}+1}$ dominates $\pi_{1:t_{0}+1}$.

\medskip
\noindent\textbf{Case 2.}
$\pi_{t_{0}}^{*} \circ \pi_{t_{0}+1}$ is popped and skipped. Then by the
pruning rule in Algorithm~2 there exists a popped and unskipped policy
$\pi_{t_{0}+1}^{*}$ that dominates
$\pi_{t_{0}}^{*} \circ \pi_{t_{0}+1}$ and satisfies
\[
V_{0}^{\pi_{t_{0}+1}^{*}}(s)
 - V_{0}^{\pi_{t_{0}}^{*} \circ \pi_{t_{0}+1}}(s)
\ge \epsilon_{t_{0}+1}.
\]
Since dominance is transitive and Lemma~4 implies that
$\pi_{t_{0}}^{*} \circ \pi_{t_{0}+1}$ dominates $\pi_{1:t_{0}+1}$, the policy
$\pi_{t_{0}+1}^{*}$ also dominates $\pi_{1:t_{0}+1}$. Moreover,
\[
\begin{aligned}
V_{0}^{\pi_{t_{0}+1}^{*}}(s) - V_{0}^{\pi_{1:t_{0}+1}}(s)
&=
\Bigl(V_{0}^{\pi_{t_{0}+1}^{*}}(s)
 - V_{0}^{\pi_{t_{0}}^{*} \circ \pi_{t_{0}+1}}(s)\Bigr)
 +
\Bigl(V_{0}^{\pi_{t_{0}}^{*} \circ \pi_{t_{0}+1}}(s)
 - V_{0}^{\pi_{1:t_{0}+1}}(s)\Bigr) \\
&\ge \epsilon_{t_{0}+1}.
\end{aligned}
\]

In either case, we obtain a popped and unskipped policy that dominates
$\pi_{1:t_{0}+1}$ and satisfies the desired value gap. By induction, this
holds up to $n=t$. Hence there exists a popped and unskipped policy
$\pi^{*}$ that dominates $\pi$ such that
\[
V_{0}^{\pi^{*}}(s) - V_{0}^{\pi}(s) \ge \epsilon_{t}.
\]
Equivalently,
\[
V_{0}^{\pi}(s) + \epsilon_{t} \le V_{0}^{\pi^{*}}(s)
\qquad \text{and} \qquad
\mathcal{G}^{\pi}(s) \subset \mathcal{G}^{\pi^{*}}(s).
\]
This proves the claim.
\end{proof}

\begin{lemma}
In General Dijkstra Search for Optimal Coverage (Algorithm~2) starting from
$s \in \mathcal{S}$, let $\pi^{*}$ be a popped and unskipped policy that is
not dominated by any previous policy in the queue. Then $\pi^{*}$ is optimal
among all policies that dominate it. Equivalently, if
$\mathcal{G} = \mathcal{G}^{\pi^{*}}(s)$, then for every
$\pi \in \Pi_{1:T}^{\mathcal{G}\subset}(s)$,
\[
V_{0}^{\pi^{*}}(s) \ge V_{0}^{\pi}(s).
\]
\end{lemma}

\begin{proof}
Fix a step $n$ at which $\pi^{*}$ is popped and unskipped. Every policy in
$\bigcup_{t=1}^{T} \Pi_{t}$ belongs to one of the following groups at step $n$:
\begin{enumerate}
    \item policies popped from $\mathcal{Q}$ at some step $\le n$,
    \item policies currently in $\mathcal{Q}$ at step $n$,
    \item policies that will enter $\mathcal{Q}$ at some step $> n$,
    \item policies that never appear in $\mathcal{Q}$.
\end{enumerate}

Because $\pi^{*}$ is the first popped and unskipped policy that is not
dominated by any previous policy in the queue, there is no policy in group (1)
whose goal set contains $\mathcal{G}^{\pi^{*}}(s)$. That is,
\[
\Bigl\{ \pi \in \bigcup_{t=1}^{T} \Pi_{t}
\,\Big|\, \mathcal{G}^{\pi^{*}}(s) \subset \mathcal{G}^{\pi}(s) \Bigr\}
\cap (1) = \emptyset.
\]

Now let
$\pi \in \bigcup_{t=1}^{T} \Pi_{t}$ satisfy
$\mathcal{G}^{\pi^{*}}(s) \subset \mathcal{G}^{\pi}(s)$. We consider the
possible groups to which $\pi$ belongs.

If $\pi$ belongs to group (2) or group (3), then Lemma~7 gives directly
\[
V_{0}^{\pi^{*}}(s) \ge V_{0}^{\pi}(s).
\]

If $\pi$ belongs to group (4), then by Lemma~10 there exists a popped and
unskipped policy $\pi'$ such that
\[
V_{0}^{\pi'}(s) \ge V_{0}^{\pi}(s)
\qquad \text{and} \qquad
\mathcal{G}^{\pi}(s) \subset \mathcal{G}^{\pi'}(s).
\]
Since $\mathcal{G}^{\pi^{*}}(s) \subset \mathcal{G}^{\pi}(s)$, we also have
\[
\mathcal{G}^{\pi^{*}}(s) \subset \mathcal{G}^{\pi'}(s).
\]
Therefore $\pi'$ cannot belong to group (1), by the defining property of
$\pi^{*}$. Hence $\pi'$ must belong to group (3), and Lemma~7 yields
\[
V_{0}^{\pi^{*}}(s) \ge V_{0}^{\pi'}(s).
\]
Combining the two inequalities gives
\[
V_{0}^{\pi^{*}}(s) \ge V_{0}^{\pi'}(s) \ge V_{0}^{\pi}(s).
\]

Thus for every policy $\pi$ with
$\mathcal{G}^{\pi^{*}}(s) \subset \mathcal{G}^{\pi}(s)$, we have
\[
V_{0}^{\pi^{*}}(s) \ge V_{0}^{\pi}(s).
\]
Equivalently, $\pi^{*}$ is the optimal goal-covering policy for the pair
$(s, \mathcal{G}^{\pi^{*}}(s))$.
\end{proof}

\section{Loss-term and search ablations}
\label{app:ablations}

We report ablations of the four-component DLR objective
(Eq.~\eqref{eq:total-loss}) and of the per-step search
(Algorithm~\ref{alg:sorl-step}). Each row gives test accuracy (\%) on the
four QA benchmarks for the five backbone models, with all other
hyperparameters held to the main-table configuration. Test split sizes match
those of Table~\ref{tab:main}.

\paragraph{Loss-term ablations.} Tables~\ref{tab:abl-alpha-abs}--\ref{tab:abl-alpha-zipf}
ablate one component of Eq.~\eqref{eq:total-loss} at a time.
\emph{Removing the policy-optimization term}
($\alpha_{\mathrm{policy}}=0$, Table~\ref{tab:abl-alpha-abs}) leaves the routing head
without supervision, and degrades GSM8K and ScienceQA most heavily.
\emph{Replacing the (generalist $+$ information-gain) pair with the specialist
conditional loss} $-\log\hat{p}_{\theta}(x\mid a)$
(Table~\ref{tab:abl-alpha-base}) drops both the unsteered LM term and the
stop-gradient log-ratio, supervising the steered model only on the
conditional likelihood. This gives the mildest degradation across the three
loss-term ablations, indicating that most of the lift can be recovered from
the conditional alone, but that explicitly contrasting against the unsteered
baseline still helps.
\emph{Removing the bigram-Zipfian regulariser} ($\alpha_{\mathrm{reg}}=0$,
Table~\ref{tab:abl-alpha-zipf}) costs $\sim 0$--$8$\,pp on ScienceQA and
StrategyQA, consistent with the prior's role of preventing code collapse.

\paragraph{Search ablations.} Tables~\ref{tab:abl-N1}--\ref{tab:abl-N8} vary the
number of rollouts $N$ used to pick the winner $a^{*}$. The main-table
configuration uses $N{=}4$; $N{=}1$ removes the per-step search entirely
(single sample, no ``select-best''), and $N{=}8$ doubles the search budget.
Tables~\ref{tab:abl-temp0}--\ref{tab:abl-temp2} change the
sampling temperature, with $\tau{=}0$ recovering the routing head's argmax
and $\tau{=}2$ flattening towards uniform.

\begin{table}[h]
\centering
\footnotesize
\caption{Ablation: $\alpha_{\mathrm{policy}}=0$ (remove the policy-optimization term). Parenthesized values are deltas (pp) vs the non-ablated DLR baseline in Table~\ref{tab:main}.}
\label{tab:abl-alpha-abs}
\begin{tabular}{lcccc}
\toprule
Model (layer) & CSQA (1221) & GSM8K (1319) & ScienceQA (2224) & StrategyQA (687) \\
\midrule
Llama-3.2-1B (L10) & $70.3_{\pm2.6}$\dlt{-1.1} & $17.4_{\pm2.0}$\dlt{-23.7} & $38.5_{\pm2.0}$\dlt{-10.6} & $47.1_{\pm3.7}$\dlt{-4.3} \\
Llama-3.2-3B (L16) & $78.9_{\pm2.3}$\dlt{-2.1} & $40.5_{\pm2.6}$\dlt{-8.6} & $56.8_{\pm2.1}$\dlt{-6.5} & $52.0_{\pm3.7}$\dlt{-10.3} \\
Qwen3-0.6B (L14)   & $63.0_{\pm2.7}$\dlt{-3.4} & $43.8_{\pm2.7}$\dlt{-5.6} & $47.7_{\pm2.1}$\dlt{-7.6} & $46.4_{\pm3.7}$\dlt{-8.2} \\
Qwen3-1.7B (L14)   & $76.1_{\pm2.4}$\dlt{-2.3} & $60.3_{\pm2.6}$\dlt{-5.4} & $56.8_{\pm2.1}$\dlt{-7.3} & $51.5_{\pm3.7}$\dlt{-8.8} \\
Qwen3-4B (L19)     & $80.6_{\pm2.2}$\dlt{-2.4} & $76.6_{\pm2.3}$\dlt{-5.5} & $59.2_{\pm2.0}$\dlt{-13.1} & $64.4_{\pm3.6}$\dlt{-4.3} \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\footnotesize
\caption{Ablation: replace the (generalist $+$ information-gain) pair in
Eq.~\eqref{eq:total-loss} with the specialist conditional loss
$-\log\hat{p}_{\theta}(x\mid a)$ (i.e.\ drop both $-\log p_{\theta}(x)$ and
the stop-gradient log-ratio, supervise only on the steered conditional).
Parenthesized values are deltas (pp) vs the non-ablated DLR baseline in
Table~\ref{tab:main}.}
\label{tab:abl-alpha-base}
\begin{tabular}{lcccc}
\toprule
Model (layer) & CSQA (1221) & GSM8K (1319) & ScienceQA (2224) & StrategyQA (687) \\
\midrule
Llama-3.2-1B (L10) & $70.4_{\pm2.6}$\dlt{-1.0} & $36.3_{\pm2.6}$\dlt{-4.8} & $42.7_{\pm2.1}$\dlt{-6.4} & $49.0_{\pm3.7}$\dlt{-2.4} \\
Llama-3.2-3B (L16) & $79.5_{\pm2.3}$\dlt{-1.5} & $44.7_{\pm2.7}$\dlt{-4.4} & $59.2_{\pm2.0}$\dlt{-4.1} & $56.0_{\pm3.7}$\dlt{-6.3} \\
Qwen3-0.6B (L14)   & $63.1_{\pm2.7}$\dlt{-3.3} & $46.2_{\pm2.7}$\dlt{-3.2} & $50.1_{\pm2.1}$\dlt{-5.2} & $51.0_{\pm3.7}$\dlt{-3.6} \\
Qwen3-1.7B (L14)   & $77.0_{\pm2.4}$\dlt{-1.4} & $62.6_{\pm2.6}$\dlt{-3.1} & $59.4_{\pm2.0}$\dlt{-4.7} & $57.0_{\pm3.7}$\dlt{-3.3} \\
Qwen3-4B (L19)     & $81.2_{\pm2.2}$\dlt{-1.8} & $77.8_{\pm2.2}$\dlt{-4.3} & $68.8_{\pm1.9}$\dlt{-3.5} & $66.0_{\pm3.5}$\dlt{-2.7} \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\footnotesize
\caption{Ablation: $\alpha_{\mathrm{reg}}=0$ (remove the bigram-Zipfian prior). Parenthesized values are deltas (pp) vs the non-ablated DLR baseline in Table~\ref{tab:main}.}
\label{tab:abl-alpha-zipf}
\begin{tabular}{lcccc}
\toprule
Model (layer) & CSQA (1221) & GSM8K (1319) & ScienceQA (2224) & StrategyQA (687) \\
\midrule
Llama-3.2-1B (L10) & $68.3_{\pm2.6}$\dlt{-3.1} & $38.1_{\pm2.6}$\dlt{-3.0} & $43.7_{\pm2.1}$\dlt{-5.4} & $48.9_{\pm3.7}$\dlt{-2.5} \\
Llama-3.2-3B (L16) & $78.5_{\pm2.3}$\dlt{-2.5} & $46.4_{\pm2.7}$\dlt{-2.7} & $59.3_{\pm2.0}$\dlt{-4.0} & $56.4_{\pm3.7}$\dlt{-5.9} \\
Qwen3-0.6B (L14)   & $62.1_{\pm2.7}$\dlt{-4.3} & $45.7_{\pm2.7}$\dlt{-3.7} & $48.2_{\pm2.1}$\dlt{-7.1} & $50.1_{\pm3.7}$\dlt{-4.5} \\
Qwen3-1.7B (L14)   & $76.9_{\pm2.4}$\dlt{-1.5} & $63.4_{\pm2.6}$\dlt{-2.3} & $60.4_{\pm2.0}$\dlt{-3.7} & $55.3_{\pm3.7}$\dlt{-5.0} \\
Qwen3-4B (L19)     & $81.2_{\pm2.2}$\dlt{-1.8} & $78.6_{\pm2.2}$\dlt{-3.5} & $64.9_{\pm2.0}$\dlt{-7.4} & $66.5_{\pm3.5}$\dlt{-2.2} \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\footnotesize
\caption{Ablation: $N{=}1$ (no search; single rollout, no ``select-best''). Parenthesized values are deltas (pp) vs the non-ablated DLR baseline in Table~\ref{tab:main}.}
\label{tab:abl-N1}
\begin{tabular}{lcccc}
\toprule
Model (layer) & CSQA (1221) & GSM8K (1319) & ScienceQA (2224) & StrategyQA (687) \\
\midrule
Llama-3.2-1B (L10) & $70.7_{\pm2.6}$\dlt{-0.7} & $37.7_{\pm2.6}$\dlt{-3.4} & $42.0_{\pm2.1}$\dlt{-7.1} & $47.8_{\pm3.7}$\dlt{-3.6} \\
Llama-3.2-3B (L16) & $78.2_{\pm2.3}$\dlt{-2.8} & $45.8_{\pm2.7}$\dlt{-3.3} & $57.2_{\pm2.1}$\dlt{-6.1} & $54.2_{\pm3.7}$\dlt{-8.1} \\
Qwen3-0.6B (L14)   & $62.4_{\pm2.7}$\dlt{-4.0} & $44.7_{\pm2.7}$\dlt{-4.7} & $47.6_{\pm2.1}$\dlt{-7.7} & $50.4_{\pm3.7}$\dlt{-4.2} \\
Qwen3-1.7B (L14)   & $75.6_{\pm2.4}$\dlt{-2.8} & $62.3_{\pm2.6}$\dlt{-3.4} & $59.1_{\pm2.0}$\dlt{-5.0} & $54.3_{\pm3.7}$\dlt{-6.0} \\
Qwen3-4B (L19)     & $81.0_{\pm2.2}$\dlt{-2.0} & $71.4_{\pm2.4}$\dlt{-10.7} & $62.5_{\pm2.0}$\dlt{-9.8} & $65.1_{\pm3.6}$\dlt{-3.6} \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\footnotesize
\caption{Ablation: $N{=}8$ (doubled search budget; baseline uses $N{=}4$). Parenthesized values are deltas (pp) vs the non-ablated DLR baseline in Table~\ref{tab:main}; doubling rollouts yields no consistent improvement over $N{=}4$.}
\label{tab:abl-N8}
\begin{tabular}{lcccc}
\toprule
Model (layer) & CSQA (1221) & GSM8K (1319) & ScienceQA (2224) & StrategyQA (687) \\
\midrule
Llama-3.2-1B (L10) & $70.4_{\pm2.6}$\dlt{-1.0} & $41.0_{\pm2.7}$\dlt{-0.1} & $48.7_{\pm2.1}$\dlt{-0.4} & $50.8_{\pm3.7}$\dlt{-0.6} \\
Llama-3.2-3B (L16) & $79.7_{\pm2.3}$\dlt{-1.3} & $49.8_{\pm2.7}$\dlt{0.7} & $63.4_{\pm2.0}$\dlt{0.1} & $62.6_{\pm3.6}$\dlt{0.3} \\
Qwen3-0.6B (L14)   & $65.1_{\pm2.7}$\dlt{-1.3} & $49.5_{\pm2.7}$\dlt{0.1} & $55.9_{\pm2.1}$\dlt{0.6} & $54.2_{\pm3.7}$\dlt{-0.4} \\
Qwen3-1.7B (L14)   & $77.9_{\pm2.3}$\dlt{-0.5} & $65.2_{\pm2.6}$\dlt{-0.5} & $63.1_{\pm2.0}$\dlt{-1.0} & $59.3_{\pm3.7}$\dlt{-1.0} \\
Qwen3-4B (L19)     & $81.3_{\pm2.2}$\dlt{-1.7} & $81.6_{\pm2.1}$\dlt{-0.5} & $72.3_{\pm1.9}$\dlt{0.0} & $68.0_{\pm3.5}$\dlt{-0.7} \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\footnotesize
\caption{Ablation: sampling temperature $\tau{=}0$ (argmax rollouts collapse to a single candidate). Parenthesized values are deltas (pp) vs the non-ablated DLR baseline in Table~\ref{tab:main}.}
\label{tab:abl-temp0}
\begin{tabular}{lcccc}
\toprule
Model (layer) & CSQA (1221) & GSM8K (1319) & ScienceQA (2224) & StrategyQA (687) \\
\midrule
Llama-3.2-1B (L10) & $67.4_{\pm2.6}$\dlt{-4.0} & $37.6_{\pm2.6}$\dlt{-3.5} & $36.5_{\pm2.0}$\dlt{-12.6} & $47.1_{\pm3.7}$\dlt{-4.3} \\
Llama-3.2-3B (L16) & $78.1_{\pm2.3}$\dlt{-2.9} & $44.8_{\pm2.7}$\dlt{-4.3} & $54.7_{\pm2.1}$\dlt{-8.6} & $51.4_{\pm3.7}$\dlt{-10.9} \\
Qwen3-0.6B (L14)   & $62.7_{\pm2.7}$\dlt{-3.7} & $45.5_{\pm2.7}$\dlt{-3.9} & $46.2_{\pm2.1}$\dlt{-9.1} & $45.8_{\pm3.7}$\dlt{-8.8} \\
Qwen3-1.7B (L14)   & $75.2_{\pm2.4}$\dlt{-3.2} & $60.1_{\pm2.6}$\dlt{-5.6} & $55.3_{\pm2.1}$\dlt{-8.8} & $49.6_{\pm3.7}$\dlt{-10.7} \\
Qwen3-4B (L19)     & $79.8_{\pm2.3}$\dlt{-3.2} & $73.9_{\pm2.4}$\dlt{-8.2} & $56.8_{\pm2.1}$\dlt{-15.5} & $62.2_{\pm3.6}$\dlt{-6.5} \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\footnotesize
\caption{Ablation: sampling temperature $\tau{=}2$ (rollouts flattened towards uniform). Parenthesized values are deltas (pp) vs the non-ablated DLR baseline in Table~\ref{tab:main}.}
\label{tab:abl-temp2}
\begin{tabular}{lcccc}
\toprule
Model (layer) & CSQA (1221) & GSM8K (1319) & ScienceQA (2224) & StrategyQA (687) \\
\midrule
Llama-3.2-1B (L10) & $70.7_{\pm2.6}$\dlt{-0.7} & $31.5_{\pm2.5}$\dlt{-9.6} & $47.8_{\pm2.1}$\dlt{-1.3} & $48.9_{\pm3.7}$\dlt{-2.5} \\
Llama-3.2-3B (L16) & $77.1_{\pm2.4}$\dlt{-3.9} & $39.3_{\pm2.6}$\dlt{-9.8} & $60.7_{\pm2.0}$\dlt{-2.6} & $59.2_{\pm3.7}$\dlt{-3.1} \\
Qwen3-0.6B (L14)   & $63.8_{\pm2.7}$\dlt{-2.6} & $42.1_{\pm2.7}$\dlt{-7.3} & $53.2_{\pm2.1}$\dlt{-2.1} & $51.6_{\pm3.7}$\dlt{-3.0} \\
Qwen3-1.7B (L14)   & $74.4_{\pm2.4}$\dlt{-4.0} & $59.6_{\pm2.6}$\dlt{-6.1} & $60.3_{\pm2.0}$\dlt{-3.8} & $57.3_{\pm3.7}$\dlt{-3.0} \\
Qwen3-4B (L19)     & $80.2_{\pm2.2}$\dlt{-2.8} & $73.9_{\pm2.4}$\dlt{-8.2} & $69.0_{\pm1.9}$\dlt{-3.3} & $66.1_{\pm3.5}$\dlt{-2.6} \\
\bottomrule
\end{tabular}
\end{table}


\section{Hyperparameter sweeps}
\label{app:hp-sweeps}

We further report full sweeps over the four non-trivial hyperparameters of the
DLR objective and of the abstraction chunking: the codebook size $C$
(Table~\ref{tab:sweep-C}), the chunk length $L$
(Table~\ref{tab:sweep-L}), the prior weight $\alpha_{\mathrm{reg}}$
(Table~\ref{tab:sweep-zipf}), and the policy-term weight
$\alpha_{\mathrm{policy}}$ (Table~\ref{tab:sweep-abs}). All sweeps use the
three smaller backbones (Llama-3.2-1B, Qwen3-0.6B, Qwen3-1.7B) across the four
QA benchmarks; all other hyperparameters are held to the main-table
configuration. We do not separately weight the information-gain term:
in Eq.~\eqref{eq:total-loss} it enters at unit weight and the functional-form
ablation (replacing the generalist $+$ info-gain pair with the specialist
loss) is reported in Table~\ref{tab:abl-alpha-base}.

\paragraph{Codebook size $C$.} On ScienceQA, increasing $C$ from $1$ to $4$
yields a $\sim 3$\,pp gain and from $4$ to $32$ a further $\sim 3$\,pp; on
GSM8K, CSQA and StrategyQA the gain is smaller but consistent. Beyond
$C{=}32$, accuracy plateaus: $C{=}64$ is within CI of $C{=}32$ on all
configurations. The main-table reports $C{=}32$ as the default.

\paragraph{Chunk length $L$.} On ScienceQA, moving from $L{=}2$ to $L{=}4$
gives $2$--$3$\,pp; $L{=}4 \to L{=}8$ is flat to slightly negative
(for $N{=}4$ rollouts). On GSM8K accuracy peaks at $L{=}4$ and degrades by
$1$--$2$\,pp at $L{=}8$. $L{=}1$ (single-token codes, no chunking) is
consistently the worst setting. The main-table reports $L{=}4$ as the default.

\paragraph{Prior weight $\alpha_{\mathrm{reg}}$.} Increasing from $0.01$ to
$0.1$ has limited effect ($\leq 1$\,pp across all settings). Further
increasing to $1.0$ tanks accuracy by $\sim 3$\,pp, consistent with the prior
becoming too strong and forcing routes towards the Zipfian reference
distribution rather than task-relevant partitions.

\paragraph{Policy weight $\alpha_{\mathrm{policy}}$.} The optimum differs by
backbone family: Qwen models prefer $\alpha_{\mathrm{policy}}{=}0.5$, while
the Llama-3.2-1B prefers a lighter $0.1$. Pushing to $\alpha_{\mathrm{policy}}{=}1.0$
tanks accuracy by $2$--$4$\,pp on Qwen and $2$--$3$\,pp on Llama, consistent
with over-regularisation of the routing head corrupting the hidden
representation (see the $\alpha_{\mathrm{policy}}{=}0$ ablation in
Table~\ref{tab:abl-alpha-abs} for the complementary direction).

\begin{table}[h]
\centering
\footnotesize
\caption{Sweep on codebook size $C$. All other hyperparameters held to the main-table configuration; $C{=}32$ is the default.}
\label{tab:sweep-C}
\begin{tabular}{llccccc}
\toprule
Model & Dataset & $C{=}1$ & $C{=}4$ & $C{=}8$ & $C{=}32$ & $C{=}64$ \\
\midrule
Qwen3-0.6B  & ScienceQA  & $48.9_{\pm2.1}$ & $51.8_{\pm2.1}$ & $53.2_{\pm2.1}$ & $55.3_{\pm2.1}$ & $55.0_{\pm2.1}$ \\
Qwen3-1.7B  & ScienceQA  & $60.0_{\pm2.0}$ & $61.3_{\pm2.0}$ & $62.5_{\pm2.0}$ & $64.1_{\pm2.0}$ & $63.9_{\pm2.0}$ \\
Llama-3.2-1B& ScienceQA  & $43.7_{\pm2.1}$ & $45.1_{\pm2.1}$ & $45.8_{\pm2.1}$ & $49.1_{\pm2.1}$ & $49.0_{\pm2.1}$ \\
\midrule
Qwen3-0.6B  & GSM8K      & $48.0_{\pm2.7}$ & $48.2_{\pm2.7}$ & $48.5_{\pm2.7}$ & $49.4_{\pm2.7}$ & $49.1_{\pm2.7}$ \\
Qwen3-1.7B  & GSM8K      & $60.0_{\pm2.6}$ & $63.8_{\pm2.6}$ & $64.5_{\pm2.6}$ & $65.7_{\pm2.6}$ & $65.3_{\pm2.6}$ \\
Llama-3.2-1B& GSM8K      & $39.0_{\pm2.6}$ & $40.2_{\pm2.6}$ & $40.9_{\pm2.7}$ & $41.4_{\pm2.7}$ & $41.1_{\pm2.7}$ \\
\midrule
Qwen3-0.6B  & CSQA       & $64.0_{\pm2.7}$ & $65.1_{\pm2.7}$ & $66.0_{\pm2.7}$ & $66.4_{\pm2.7}$ & $66.2_{\pm2.7}$ \\
Qwen3-1.7B  & CSQA       & $75.2_{\pm2.4}$ & $77.1_{\pm2.4}$ & $78.0_{\pm2.3}$ & $78.4_{\pm2.3}$ & $78.1_{\pm2.3}$ \\
Llama-3.2-1B& CSQA       & $69.1_{\pm2.6}$ & $70.2_{\pm2.6}$ & $71.0_{\pm2.5}$ & $71.4_{\pm2.5}$ & $71.2_{\pm2.5}$ \\
\midrule
Qwen3-0.6B  & StrategyQA & $51.2_{\pm3.7}$ & $53.0_{\pm3.7}$ & $54.0_{\pm3.7}$ & $54.6_{\pm3.7}$ & $54.3_{\pm3.7}$ \\
Qwen3-1.7B  & StrategyQA & $57.3_{\pm3.7}$ & $59.0_{\pm3.7}$ & $59.4_{\pm3.7}$ & $60.3_{\pm3.7}$ & $60.1_{\pm3.7}$ \\
Llama-3.2-1B& StrategyQA & $48.2_{\pm3.7}$ & $49.3_{\pm3.7}$ & $50.4_{\pm3.7}$ & $51.4_{\pm3.7}$ & $51.1_{\pm3.7}$ \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\footnotesize
\caption{Sweep on chunk length $L$. $L{=}4$ is the default.}
\label{tab:sweep-L}
\begin{tabular}{llcccc}
\toprule
Model & Dataset & $L{=}1$ & $L{=}2$ & $L{=}4$ & $L{=}8$ \\
\midrule
Qwen3-0.6B  & ScienceQA  & $42.3_{\pm2.1}$ & $53.9_{\pm2.1}$ & $55.3_{\pm2.1}$ & $53.2_{\pm2.1}$ \\
Qwen3-1.7B  & ScienceQA  & $61.7_{\pm2.0}$ & $63.0_{\pm2.0}$ & $64.1_{\pm2.0}$ & $62.8_{\pm2.0}$ \\
Llama-3.2-1B& ScienceQA  & $43.1_{\pm2.1}$ & $44.5_{\pm2.1}$ & $49.1_{\pm2.1}$ & $47.4_{\pm2.1}$ \\
\midrule
Qwen3-0.6B  & GSM8K      & $46.6_{\pm2.7}$ & $47.7_{\pm2.7}$ & $49.4_{\pm2.7}$ & $47.2_{\pm2.7}$ \\
Qwen3-1.7B  & GSM8K      & $61.3_{\pm2.6}$ & $63.0_{\pm2.6}$ & $65.7_{\pm2.6}$ & $64.5_{\pm2.6}$ \\
Llama-3.2-1B& GSM8K      & $40.1_{\pm2.6}$ & $40.9_{\pm2.7}$ & $41.4_{\pm2.7}$ & $40.8_{\pm2.7}$ \\
\midrule
Qwen3-0.6B  & CSQA       & $64.0_{\pm2.7}$ & $66.7_{\pm2.6}$ & $66.4_{\pm2.7}$ & $65.3_{\pm2.7}$ \\
Qwen3-1.7B  & CSQA       & $75.2_{\pm2.4}$ & $76.1_{\pm2.4}$ & $78.4_{\pm2.3}$ & $76.5_{\pm2.4}$ \\
Llama-3.2-1B& CSQA       & $70.5_{\pm2.6}$ & $70.9_{\pm2.5}$ & $71.4_{\pm2.5}$ & $71.2_{\pm2.5}$ \\
\midrule
Qwen3-0.6B  & StrategyQA & $50.6_{\pm3.7}$ & $53.2_{\pm3.7}$ & $54.6_{\pm3.7}$ & $52.0_{\pm3.7}$ \\
Qwen3-1.7B  & StrategyQA & $58.4_{\pm3.7}$ & $58.5_{\pm3.7}$ & $60.3_{\pm3.7}$ & $59.7_{\pm3.7}$ \\
Llama-3.2-1B& StrategyQA & $49.1_{\pm3.7}$ & $50.8_{\pm3.7}$ & $51.4_{\pm3.7}$ & $51.9_{\pm3.7}$ \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\footnotesize
\caption{Sweep on $\alpha_{\mathrm{reg}}$ (bigram-Zipfian prior weight). $\alpha_{\mathrm{reg}}{=}0.1$ is the default; cf.\ the $\alpha_{\mathrm{reg}}{=}0$ ablation in Table~\ref{tab:abl-alpha-zipf}.}
\label{tab:sweep-zipf}
\begin{tabular}{llccc}
\toprule
Model & Dataset & $\alpha_{\mathrm{reg}}{=}0.01$ & $\alpha_{\mathrm{reg}}{=}0.1$ & $\alpha_{\mathrm{reg}}{=}1.0$ \\
\midrule
Qwen3-0.6B  & ScienceQA  & $54.3_{\pm2.1}$ & $55.0_{\pm2.1}$ & $51.1_{\pm2.1}$ \\
Qwen3-1.7B  & ScienceQA  & $63.7_{\pm2.0}$ & $64.1_{\pm2.0}$ & $61.9_{\pm2.0}$ \\
Llama-3.2-1B& ScienceQA  & $49.0_{\pm2.1}$ & $49.1_{\pm2.1}$ & $45.0_{\pm2.1}$ \\
\midrule
Qwen3-0.6B  & GSM8K      & $48.4_{\pm2.7}$ & $49.4_{\pm2.7}$ & $46.5_{\pm2.7}$ \\
Qwen3-1.7B  & GSM8K      & $64.6_{\pm2.6}$ & $65.7_{\pm2.6}$ & $60.2_{\pm2.6}$ \\
Llama-3.2-1B& GSM8K      & $40.3_{\pm2.7}$ & $41.3_{\pm2.7}$ & $37.8_{\pm2.6}$ \\
\midrule
Qwen3-0.6B  & CSQA       & $66.1_{\pm2.7}$ & $66.4_{\pm2.7}$ & $62.3_{\pm2.7}$ \\
Qwen3-1.7B  & CSQA       & $78.5_{\pm2.3}$ & $78.4_{\pm2.3}$ & $74.9_{\pm2.4}$ \\
Llama-3.2-1B& CSQA       & $70.6_{\pm2.6}$ & $71.4_{\pm2.5}$ & $66.0_{\pm2.7}$ \\
\midrule
Qwen3-0.6B  & StrategyQA & $53.3_{\pm3.7}$ & $54.6_{\pm3.7}$ & $50.2_{\pm3.7}$ \\
Qwen3-1.7B  & StrategyQA & $58.8_{\pm3.7}$ & $60.3_{\pm3.7}$ & $55.0_{\pm3.7}$ \\
Llama-3.2-1B& StrategyQA & $50.6_{\pm3.7}$ & $51.4_{\pm3.7}$ & $46.9_{\pm3.7}$ \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\footnotesize
\caption{Sweep on $\alpha_{\mathrm{policy}}$ (policy-term weight). The default is family-dependent: $\alpha_{\mathrm{policy}}{=}0.5$ for Qwen and $\alpha_{\mathrm{policy}}{=}0.1$ for Llama (best per row \textbf{bold}); cf.\ the $\alpha_{\mathrm{policy}}{=}0$ ablation in Table~\ref{tab:abl-alpha-abs}.}
\label{tab:sweep-abs}
\begin{tabular}{llcccc}
\toprule
Model & Dataset & $\alpha_{\mathrm{policy}}{=}0.01$ & $\alpha_{\mathrm{policy}}{=}0.1$ & $\alpha_{\mathrm{policy}}{=}0.5$ & $\alpha_{\mathrm{policy}}{=}1.0$ \\
\midrule
Qwen3-0.6B  & ScienceQA  & $50.7_{\pm2.1}$ & $53.5_{\pm2.1}$ & $\mathbf{55.8_{\pm2.1}}$ & $52.9_{\pm2.1}$ \\
Qwen3-1.7B  & ScienceQA  & $61.0_{\pm2.0}$ & $63.6_{\pm2.0}$ & $\mathbf{64.3_{\pm2.0}}$ & $60.7_{\pm2.0}$ \\
Llama-3.2-1B& ScienceQA  & $47.4_{\pm2.1}$ & $\mathbf{49.1_{\pm2.1}}$ & $47.2_{\pm2.1}$ & $46.2_{\pm2.1}$ \\
\midrule
Qwen3-0.6B  & GSM8K      & $47.6_{\pm2.7}$ & $48.9_{\pm2.7}$ & $\mathbf{49.4_{\pm2.7}}$ & $45.7_{\pm2.7}$ \\
Qwen3-1.7B  & GSM8K      & $63.0_{\pm2.6}$ & $63.2_{\pm2.6}$ & $\mathbf{65.7_{\pm2.6}}$ & $62.1_{\pm2.6}$ \\
Llama-3.2-1B& GSM8K      & $39.4_{\pm2.6}$ & $\mathbf{41.3_{\pm2.7}}$ & $40.5_{\pm2.7}$ & $38.2_{\pm2.6}$ \\
\midrule
Qwen3-0.6B  & CSQA       & $65.1_{\pm2.7}$ & $65.5_{\pm2.7}$ & $\mathbf{66.4_{\pm2.7}}$ & $62.7_{\pm2.7}$ \\
Qwen3-1.7B  & CSQA       & $76.3_{\pm2.4}$ & $77.8_{\pm2.3}$ & $\mathbf{78.4_{\pm2.3}}$ & $73.9_{\pm2.5}$ \\
Llama-3.2-1B& CSQA       & $70.2_{\pm2.6}$ & $\mathbf{71.4_{\pm2.5}}$ & $71.6_{\pm2.5}$ & $69.0_{\pm2.6}$ \\
\midrule
Qwen3-0.6B  & StrategyQA & $51.8_{\pm3.7}$ & $52.2_{\pm3.7}$ & $\mathbf{54.6_{\pm3.7}}$ & $50.5_{\pm3.7}$ \\
Qwen3-1.7B  & StrategyQA & $58.7_{\pm3.7}$ & $59.9_{\pm3.7}$ & $\mathbf{60.3_{\pm3.7}}$ & $56.2_{\pm3.7}$ \\
Llama-3.2-1B& StrategyQA & $49.1_{\pm3.7}$ & $\mathbf{51.4_{\pm3.7}}$ & $50.0_{\pm3.7}$ & $47.1_{\pm3.7}$ \\
\bottomrule
\end{tabular}
\end{table}




% \section*{NeurIPS Paper Checklist}

% %%% BEGIN INSTRUCTIONS %%%
% The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: {\bf The papers not including the checklist will be desk rejected.} The checklist should follow the references and follow the (optional) supplemental material.  The checklist does NOT count towards the page
% limit. 

% Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:
% \begin{itemize}
%     \item You should answer \answerYes{}, \answerNo{}, or \answerNA{}.
%     \item \answerNA{} means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
%     \item Please provide a short (1–2 sentence) justification right after your answer (even for NA). 
%    % \item {\bf The papers not including the checklist will be desk rejected.}
% \end{itemize}

% {\bf The checklist answers are an integral part of your paper submission.} They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.

% The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "\answerYes{}" is generally preferable to "\answerNo{}", it is perfectly acceptable to answer "\answerNo{}" provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "\answerNo{}" or "\answerNA{}" is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer \answerYes{} to a question, in the justification please point to the section(s) where related material for the question can be found.

% IMPORTANT, please:
% \begin{itemize}
%     \item {\bf Delete this instruction block, but keep the section heading ``NeurIPS paper checklist"},
%     \item  {\bf Keep the checklist subsection headings, questions/answers and guidelines below.}
%     \item {\bf Do not modify the questions and only use the provided macros for your answers}.
% \end{itemize} 
 

% %%% END INSTRUCTIONS %%%


% \begin{enumerate}

% \item {\bf Claims}
%     \item[] Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?
%     \item[] Answer: \answerTODO{} % Replace by \answerYes{}, \answerNo{}, or \answerNA{}.
%     \item[] Justification: \justificationTODO{}
%     \item[] Guidelines:
%     \begin{itemize}
%         \item The answer NA means that the abstract and introduction do not include the claims made in the paper.
%         \item The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers. 
%         \item The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings. 
%         \item It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper. 
%     \end{itemize}

% \item {\bf Limitations}
%     \item[] Question: Does the paper discuss the limitations of the work performed by the authors?
%     \item[] Answer: \answerTODO{} % Replace by \answerYes{}, \answerNo{}, or \answerNA{}.
%     \item[] Justification: \justificationTODO{}
%     \item[] Guidelines:
%     \begin{itemize}
%         \item The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper. 
%         \item The authors are encouraged to create a separate "Limitations" section in their paper.
%         \item The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
%         \item The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
%         \item The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
%         \item The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
%         \item If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
%         \item While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
%     \end{itemize}

% \item {\bf Theory Assumptions and Proofs}
%     \item[] Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
%     \item[] Answer: \answerTODO{} % Replace by \answerYes{}, \answerNo{}, or \answerNA{}.
%     \item[] Justification: \justificationTODO{}
%     \item[] Guidelines:
%     \begin{itemize}
%         \item The answer NA means that the paper does not include theoretical results. 
%         \item All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.
%         \item All assumptions should be clearly stated or referenced in the statement of any theorems.
%         \item The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition. 
%         \item Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
%         \item Theorems and Lemmas that the proof relies upon should be properly referenced. 
%     \end{itemize}

%     \item {\bf Experimental Result Reproducibility}
%     \item[] Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?
%     \item[] Answer: \answerTODO{} % Replace by \answerYes{}, \answerNo{}, or \answerNA{}.
%     \item[] Justification: \justificationTODO{}
%     \item[] Guidelines:
%     \begin{itemize}
%         \item The answer NA means that the paper does not include experiments.
%         \item If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
%         \item If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable. 
%         \item Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
%         \item While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
%         \begin{enumerate}
%             \item If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
%             \item If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
%             \item If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
%             \item We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
%         \end{enumerate}
%     \end{itemize}


% \item {\bf Open access to data and code}
%     \item[] Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?
%     \item[] Answer: \answerTODO{} % Replace by \answerYes{}, \answerNo{}, or \answerNA{}.
%     \item[] Justification: \justificationTODO{}
%     \item[] Guidelines:
%     \begin{itemize}
%         \item The answer NA means that paper does not include experiments requiring code.
%         \item Please see the NeurIPS code and data submission guidelines (\url{https://nips.cc/public/guides/CodeSubmissionPolicy}) for more details.
%         \item While we encourage the release of code and data, we understand that this might not be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
%         \item The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (\url{https://nips.cc/public/guides/CodeSubmissionPolicy}) for more details.
%         \item The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
%         \item The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
%         \item At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
%         \item Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.
%     \end{itemize}


% \item {\bf Experimental Setting/Details}
%     \item[] Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
%     \item[] Answer: \answerTODO{} % Replace by \answerYes{}, \answerNo{}, or \answerNA{}.
%     \item[] Justification: \justificationTODO{}
%     \item[] Guidelines:
%     \begin{itemize}
%         \item The answer NA means that the paper does not include experiments.
%         \item The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
%         \item The full details can be provided either with the code, in appendix, or as supplemental material.
%     \end{itemize}

% \item {\bf Experiment Statistical Significance}
%     \item[] Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
%     \item[] Answer: \answerTODO{} % Replace by \answerYes{}, \answerNo{}, or \answerNA{}.
%     \item[] Justification: \justificationTODO{}
%     \item[] Guidelines:
%     \begin{itemize}
%         \item The answer NA means that the paper does not include experiments.
%         \item The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
%         \item The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
%         \item The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
%         \item The assumptions made should be given (e.g., Normally distributed errors).
%         \item It should be clear whether the error bar is the standard deviation or the standard error of the mean.
%         \item It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96\% CI, if the hypothesis of Normality of errors is not verified.
%         \item For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
%         \item If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.
%     \end{itemize}

% \item {\bf Experiments Compute Resources}
%     \item[] Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
%     \item[] Answer: \answerTODO{} % Replace by \answerYes{}, \answerNo{}, or \answerNA{}.
%     \item[] Justification: \justificationTODO{}
%     \item[] Guidelines:
%     \begin{itemize}
%         \item The answer NA means that the paper does not include experiments.
%         \item The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
%         \item The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute. 
%         \item The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). 
%     \end{itemize}
    
% \item {\bf Code Of Ethics}
%     \item[] Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics \url{https://neurips.cc/public/EthicsGuidelines}?
%     \item[] Answer: \answerTODO{} % Replace by \answerYes{}, \answerNo{}, or \answerNA{}.
%     \item[] Justification: \justificationTODO{}
%     \item[] Guidelines:
%     \begin{itemize}
%         \item The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
%         \ite\textbf{}m If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
%         \item The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
%     \end{itemize}


% \item {\bf Broader Impacts}
%     \item[] Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
%     \item[] Answer: \answerTODO{} % Replace by \answerYes{}, \answerNo{}, or \answerNA{}.
%     \item[] Justification: \justificationTODO{}
%     \item[] Guidelines:
%     \begin{itemize}
%         \item The answer NA means that there is no societal impact of the work performed.
%         \item If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
%         \item Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
%         \item The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
%         \item The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
%         \item If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
%     \end{itemize}
    
% \item {\bf Safeguards}
%     \item[] Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
%     \item[] Answer: \answerTODO{} % Replace by \answerYes{}, \answerNo{}, or \answerNA{}.
%     \item[] Justification: \justificationTODO{}
%     \item[] Guidelines:
%     \begin{itemize}
%         \item The answer NA means that the paper poses no such risks.
%         \item Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters. 
%         \item Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
%         \item We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
%     \end{itemize}

% \item {\bf Licenses for existing assets}
%     \item[] Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
%     \item[] Answer: \answerTODO{} % Replace by \answerYes{}, \answerNo{}, or \answerNA{}.
%     \item[] Justification: \justificationTODO{}
%     \item[] Guidelines:
%     \begin{itemize}
%         \item The answer NA means that the paper does not use existing assets.
%         \item The authors should cite the original paper that produced the code package or dataset.
%         \item The authors should state which version of the asset is used and, if possible, include a URL.
%         \item The name of the license (e.g., CC-BY 4.0) should be included for each asset.
%         \item For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
%         \item If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, \url{paperswithcode.com/datasets} has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
%         \item For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
%         \item If this information is not available online, the authors are encouraged to reach out to the asset's creators.
%     \end{itemize}

% \item {\bf New Assets}
%     \item[] Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
%     \item[] Answer: \answerTODO{} % Replace by \answerYes{}, \answerNo{}, or \answerNA{}.
%     \item[] Justification: \justificationTODO{}
%     \item[] Guidelines:
%     \begin{itemize}
%         \item The answer NA means that the paper does not release new assets.
%         \item Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc. 
%         \item The paper should discuss whether and how consent was obtained from people whose asset is used.
%         \item At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
%     \end{itemize}

% \item {\bf Crowdsourcing and Research with Human Subjects}
%     \item[] Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? 
%     \item[] Answer: \answerTODO{} % Replace by \answerYes{}, \answerNo{}, or \answerNA{}.
%     \item[] Justification: \justificationTODO{}
%     \item[] Guidelines:
%     \begin{itemize}
%         \item The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
%         \item Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. 
%         \item According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 
%     \end{itemize}

% \item {\bf Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects}
%     \item[] Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?
%     \item[] Answer: \answerTODO{} % Replace by \answerYes{}, \answerNo{}, or \answerNA{}.
%     \item[] Justification: \justificationTODO{}
%     \item[] Guidelines:
%     \begin{itemize}
%         \item The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
%         \item Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. 
%         \item We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. 
%         \item For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.
%     \end{itemize}

% \end{enumerate}



% \begin{algorithm}[H]
% \caption{Self-Organizing RL (No Composition Ver.)}
% \KwIn{Environment, universal Q-function $Q_{\theta}(s, a, z)$, goal encoder $\phi : \mathcal{S} \to \mathbb{R}^D$, vocabulary matrix $M \in \mathbb{R}^{N \times D}$, unroll steps $T$}
% \KwOut{Learned embeddings $M = \{M_1, \ldots, M_N\}$, universal Q-function $Q_{\theta}$, optimal concatenated policies}

% Initialize priority queue $Q = \{( \emptyset , 0, s_0, 0)\}$, max values $V_{\max}[j] \gets -\infty$ for $j = 1, \ldots, N$\;

% \While{$Q \neq \emptyset$}{
%     $(\tau'_{1:n}, v', s', t') \gets \text{pop}(Q)$\;

%     \For{$j = 1$ \KwTo $N$}{
%         $\tau_{1:n+1} \gets \tau'_{1:n} \circ j$\;
%         $s_{0} \gets s', \quad v \gets v', \quad \hat{\gamma} \gets \gamma^{t'}$\;

%         \tcp{Exploration stage}
%         \For{$t = 0$ \KwTo $T-1$}{
%             $a_{t} \sim \arg\max_a Q(s_{t}, a, M_j)$\;
%             Sample $s_{t+1} \sim p(\cdot | s_{t}, a_{t})$\;
%             $v \gets v + \hat{\gamma} \cdot \gamma^{t} r_{t}$\;
%         }
        
%         \tcp{Composition stage}
%         $M_{j} \gets \phi(s_{T})$\;\tcp*{Map to $M_j$}
%         $V_{\max}[j] \gets \max(V_{\max}[j], v)$\;
%         Update $Q(\cdot, \cdot, M_j)$ to approximate $v$ on $\tau$\;
%         Update $M_j$ and $\phi$\;

%         Push $(\tau_{1:n+1}, v, s_{T}, t_{curr} + T )$ to $Q$\;
%     }
% }
% \end{algorithm}


\end{document}
