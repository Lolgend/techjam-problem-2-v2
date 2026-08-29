

# **MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement** 

**Jaehyun Nam**<sup>1 2 *</sup> **, Jinsung Yoon**<sup>1</sup> **, Jiefeng Chen**<sup>1</sup> **, Jinwoo Shin**<sup>2</sup> **, Sercan Ö. Arık**<sup>1</sup> **and Tomas Pfister**<sup>1</sup> 1Google Cloud, 2KAIST 

**Agents based on large language models (LLMs) for machine learning engineering (MLE) can automatically implement ML models via code generation. However, existing approaches to build such agents often rely heavily on inherent LLM knowledge and employ coarse exploration strategies that modify the entire code structure at once. This limits their ability to select effective task-specific models and perform deep exploration within specific components, such as experimenting extensively with feature engineering options. To overcome these, we propose** **_MLE-STAR_ , a novel approach to build MLE agents. MLESTAR first leverages external knowledge by using a search engine to retrieve effective models from the web, forming an initial solution, then iteratively refines it by exploring various strategies targeting specific ML components. This exploration is guided by ablation studies analyzing the impact of individual code blocks. Furthermore, we introduce a novel ensembling method using an effective strategy suggested by MLE-STAR. Our experimental results show that MLE-STAR achieves medals in 64% of the Kaggle competitions on the MLE-bench Lite, significantly outperforming the best alternative.** 

### **1. Introduction** 

The proliferation of machine learning (ML) has driven high-performance applications across diverse real-world scenarios, from fundamental tasks like tabular classification (Chen and Guestrin, 2016; Hollmann et al., 2025; Prokhorenkova et al., 2018) to complex ones such as image denoising (Fan et al., 2019). Despite these advances, developing such models remains a labor-intensive process for data scientists, involving extensive iterative experimentation and data engineering (Hollmann et al., 2023; Nam et al., 2024). To streamline such intensive workflows, recent research has focused on employing large language models (LLMs) (Brown et al., 2020; Team et al., 2024; Touvron et al., 2023) as _machine learning engineering (MLE) agents_ (Guo et al., 2024; Hong et al., 2024; Jiang et al., 2025). By harnessing the coding and reasoning capabilities inherent in LLMs (Jain et al., 2025; Jimenez et al., 2024), these agents conceptualize ML tasks as code optimization problems. They then navigate the potential code solutions ultimately producing executable code ( _e.g._ , a Python script) based on a provided task description and dataset (see Figure 1). 

Despite their promise as pioneering efforts, current MLE agents face several obstacles that limit their effectiveness. First, due to their strong reliance on inherent LLM knowledge, they are often biased toward familiar and frequently used methods ( _e.g._ , the scikit-learn library (Pedregosa et al., 2011) for tabular data), neglecting potentially promising task-specific methods. Additionally, these agents (Guo et al., 2024; Jiang et al., 2025) typically employ an exploration strategy that modifies the entire code structure at once in each iteration. This often results in agents pivoting prematurely to other steps ( _e.g._ , model selection or hyperparameter tuning) because they lack the ability to perform deep, iterative exploration within specific pipeline components, such as experimenting different feature engineering options extensively. 

**Contributions.** We propose **MLE-STAR** , a novel **ML E** ngineering agent that integrates web **S** earch and **TA** rgeted code block **R** efinement (see Figure 2 for an overview). Specifically, generating initial solution code, MLE-STAR utilizes Google Search to retrieve relevant and potentially state-of-the-art 

_Corresponding author(s): jaehyun.nam@kaist.ac.kr, jinsungyoon@google.com_ * This work was done while Jaehyun was a student researcher at Google Cloud. 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 



<!-- Start of picture text -->
Planning<br>Input Training<br>Debugging<br>MLE Agent<br>…<br><!-- End of picture text -->



<!-- Start of picture text -->
Task Description<br>Dataset<br>Machine Learning Tasks<br><!-- End of picture text -->



<!-- Start of picture text -->
Outputputut<br>Solution Script<br><!-- End of picture text -->



<!-- Start of picture text -->
Training Outputputut<br>Debugging<br>…<br><!-- End of picture text -->

Figure 1 | **Problem setup.** ML Engineering agents are designed to process a task description and datasets across various modalities ( _e.g._ , tabular, text, image, audio, etc.) with the objective of determining the optimal solution for a given machine learning problem, such as classification, regression, sequence-to-sequence generation, image denoising, text normalization, etc. 

approaches that could be effective towards building a model. Moreover, to improve the solution, MLE-STAR extracts a specific code block that represents a distinct ML pipeline component, such as feature engineering or ensemble building, and then concentrates on exploring strategies that are targeted to that component, using previous attempts as feedback to reflect on. Here, to identify the code block that has the greatest impact on performance, MLE-STAR performs an ablation study that evaluates the contribution of each ML component. This refinement process is repeated, modifying various code blocks ( _i.e._ , other ML components). In addition, we introduce a novel method to generate ensembles. MLE-STAR first proposes multiple candidate solutions. Then, instead of relying on a simple voting based on validation scores, MLE-STAR merges these candidates into a single improved solution using an ensemble strategy proposed by the agent itself. This ensemble strategy is iteratively refined based on the performance of the previous strategies. 

To verify the effectiveness, we conduct comprehensive evaluations of MLE-STAR using the MLEbench’s Kaggle competitions (Chan et al., 2025). The experimental results demonstrate that MLE-STAR, requiring only minimal human effort ( _e.g._ , defining initial prompts that are generalizable to any tasks), significantly outperforms previous methods (Jiang et al., 2025), including those requiring manual labor to collect strategies from Kaggle (Guo et al., 2024). In particular, MLE-STAR achieves a substantial gain in medal achievement, improving it from 25.8% to 63.6% when compared to the top-performing baseline. Additionally, we show that our proposed ensemble technique provides a meaningful improvement to MLE-STAR. 

### **2. Related work** 

**LLM agents.** Recent advances in LLMs have led to an active research in autonomous agents. Generalpurpose agents like ReAct (Yao et al., 2023) and HuggingGPT (Shen et al., 2023) typically use external tools to analyze various problems. Specialized agents, such as Voyager (Wang et al., 2023) for Minecraft or AlphaCode (Li et al., 2022) for code generation, excel in specific domains, often using execution feedback to iteratively improve their approach. Extending these, we introduce MLE-STAR, an LLM agent that specialized in ML tasks. 

**Automated machine learning.** Automated machine learning (AutoML) aims to reduce reliance on human experts by automating end-to-end ML pipelines (Feurer et al., 2022; Jin et al., 2019; LeDell and Poirier, 2020). Auto-WEKA (Kotthoff et al., 2017), TPOT (Olson and Moore, 2016), and recent advances such as AutoGluon (Erickson et al., 2020), have made progress through exploring within predefined model or hyperparameter spaces. AutoML research also specializes in areas such as neural network design (Elsken et al., 2019; Pham et al., 2018; Real et al., 2019; Zoph and Le, 2017), and feature engineering (Fan et al., 2010; Horn et al., 2019; Kanter and Veeramachaneni, 2015; Li et al., 2023; Zhang et al., 2023). However, these methods rely on predefined search spaces, which often 

2 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

require domain expertise to define. To address this, LLM-based MLE agents (Guo et al., 2024; Jiang et al., 2025), including MLE-STAR, are emerging, since they employ effective exploration strategies directly in the code space, without the need of manually-curated search spaces. 

**MLE agents.** Leveraging coding and reasoning capabilities of LLMs (Jain et al., 2025; Jimenez et al., 2024), research has been conducted on use of LLMs as MLE agents (Hong et al., 2024; Li et al., 2024; Schmidgall et al., 2025), which generate solution code, to automate ML workflows. While MLAB (Huang et al., 2024a) and OpenHands (Wang et al., 2024) take general actions by calling tools to perform ML tasks, several studies specialize in ML automation. AIDE (Jiang et al., 2025) generates candidate solutions in a tree structure to facilitate code space exploration. However, its heavy reliance on the LLM’s internal knowledge can lead to outdated or overly simple model choices, and its refinement may prematurely shift focus between pipeline stages. DS-Agent (Guo et al., 2024) uses case-based reasoning (Kolodner, 1992; Watson and Marir, 1994) to discover strategies for solution generation by utilizing manually curated cases (primarily from Kaggle). However, DS-Agent suffers from scalability issues due to its reliance on a manually built case bank, which requires significant human effort and can lead to solutions that are overfit to the source patterns. Also, it restricts applicability to novel task types (like complex multi-modal problems). Our method addresses these limitations. Instead of attempting to explore the broader code space or relying on a static case bank, MLE-STAR strategically explores implementation options for specific ML pipeline components. It also improves scalability by using LLMs with search as tool to retrieve effective models that fit the task beyond the constraints of a fixed case bank. 

### **3. MLE-STAR** 

We introduce the proposed framework for MLE agents, MLE-STAR, that effectively leverages the coding and reasoning capabilities of LLMs to solve ML tasks. In a nutshell, our approach is based on first generating an initial solution by using web search as a tool (Section 3.1), and then refining solutions via nested loops. The outer loop targets one code block, which corresponds to the specific ML component extracted through an ablation study. The inner loop iteratively refines _only_ this block until the outer loop moves to the next target (Section 3.2). We propose a novel ensemble method that improves the performance using the plan proposed by LLMs, which is iteratively refined (Section 3.3). To mitigate potential undesirable behaviors from LLMs, such as using test sample statistics for missing value imputation, we introduce specific modules (detailed in Section 3.4). The prompts and algorithms used in each step can be found in Appendix A and B, respectively. 

**Problem setup.** Formally, our goal is to find an optimal solution _𝑠_<sup>∗</sup> = arg max _𝑠_ ∈S _ℎ_ ( _𝑠_ ), where S is the space of possible solutions ( _i.e._ , Python scripts) and _ℎ_ : S → ℝ is a score function ( _e.g._ , validation accuracy) (Jiang et al., 2025). To obtain _𝑠_<sup>∗</sup> , we propose a multi-agent framework A, which takes datasets D (that might contain multiple files) and a task description T `task` (which includes task types, data modalities, score functions, etc.) as input.<sup>1</sup> Here, A consists of _𝑛_ LLM agents (A1 _,_ · · · _,_ A _𝑛_ ). Each agent A _𝑖_ possesses specific functionalities, which are elaborated upon in following sections. 

#### **3.1. Generating an initial solution using web search as a tool** 

**Candidate model search.** MLE-STAR starts by generating an initial solution. For high performance in ML tasks, selecting the appropriate model is paramount. However, relying solely on an LLM for model suggestions can lead to suboptimal choices. For instance, we observe that LLMs propose models 

> 1MLE-STAR works across any data modalities ( _e.g._ , tabular, image, text, audio) and task types ( _e.g._ , classification, image-to-image, sequence-to-sequence) – it is not restricted to specific inputs or objectives. 

3 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 



<!-- Start of picture text -->
What models Retrieved information Evaluation Val. Score Initial<br>Solution<br>are effective?<br>0.91<br>{Model description} 1st Integrate<br>Search as tool {Example code} Sort<br>0.92<br>0.90<br>Task description Generate Python script<br>0.89<br>MLE-STAR w/ Dataset 0.85<br>2nd Integrate Discard<br>(a) Initialization<br>What component  Python script for Execution result Imputation method impacts the most,<br>impacts the most? ablation study and was not refined before!<br>Let’s refine the encoding strategy.<br>Modify specific<br>components Original: 0.93Median imputing: 0.95 numeric_transformer   ('imputer', SimpleImputer= Pipeline(strategy(steps=[='mean')),<br>Previous solution Removing feature 0.94    ('scaler', StandardScaler())])<br>MLE-STAR w/ MLE-STAR w/<br>Extracted code block<br>(b) Target code block extraction<br>Outer Loop<br>Suggest plan Implement plan k Refined<br>How about Plan k?<br>Target code block Code Block Improved<br>Evaluation Solution<br>Plan 1 → Score 1<br>Target Plan 2 → Score 2 Replace the code<br>Code Block … Target Replace Refined block when done<br>Plan k → Score k Code Block Code Block Refined<br>MLE-STAR w/ Update trajectory &  Code Block<br>Feedback Implement<br>(c) Code block refinement best plan<br><!-- End of picture text -->



<!-- Start of picture text -->
MLE-STAR w/<br><!-- End of picture text -->



<!-- Start of picture text -->
MLE-STAR w/<br><!-- End of picture text -->

Figure 2 | **Overview of MLE-STAR.** (a) Using search as a tool, MLE-STAR retrieves task-specific models and uses them to generate an initial solution. (b) In each refinement step, MLE-STAR performs an ablation study to extract the code block that have the greatest impact. Previously modified code blocks are also provided as feedback for diversity. (c) The extracted code block is iteratively refined based on plans suggested by the LLM, which explores various plans using previous experiments as feedback ( _i.e._ , inner loop), and the target code block is also selected repeatedly ( _i.e._ , outer loop, where the improved solution of (c) becomes the previous solution in (b)). 

like logistic regression (Pedregosa et al., 2011) even for competitions like jigsaw-toxic-commentclassification, which is a text classification task, potentially because LLMs favor familiar patterns from their pre-training data over up-to-date information. To mitigate this, we propose using web search as a tool for MLE-STAR first to retrieve _𝑀_ effective, state-of-the-art models for the given task. This retrieved context is then used to guide the LLM in generating a more informed initial solution. Formally: 



where T `model` represents the description of a retrieved model, while T `code` provides corresponding example code. This example code is needed since the LLM can be unfamiliar with the model and cannot generate the executable code without proper guidance. Then, MLE-STAR involves evaluating of the performance of model _𝑖_ . To achieve this, candidate evaluation agent A `init` first generates code, _𝑠_<sup>_𝑖_This process is formally defined as:</sup> `init`<sup>, using the retrieved model to solve the given ML task.</sup> 



We evaluate the performance of each _𝑠_ using a task-specific metric _ℎ_ on dataset D. We denote the resulting score by _ℎ_ ( _𝑠_ ), which encapsulates the entire process done in _𝑠_ : splitting D into training and validation sets, training the model specified in _𝑠_ using the training data, and calculating _ℎ_ on 

4 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

the validation data. The performance for _𝑠_ `init`<sup>_𝑖_isthus</sup><sup>_ℎ_(</sup><sup>_𝑠_</sup> `init`<sup>_𝑖_).Asaresult,asetofcodescripts</sup> S `init` = { _𝑠_ `init`<sup>1</sup><sup>_,_· · ·</sup><sup>_, 𝑠_</sup> `init`<sup>_𝑀_}and their performance scores{</sup><sup>_ℎ_(</sup><sup>_𝑠_</sup> `init`<sup>1)</sup><sup>_,_· · ·</sup><sup>_, ℎ_(</sup><sup>_𝑠_</sup> `init`<sup>_𝑀_)}are obtained.</sup> **Merging candidate models for initial solution.** After the evaluation of the _𝑀_ retrieved models, a consolidated initial solution _𝑠_ 0 is constructed through an iterative merging procedure. Specifically, we first define _𝜋_ be a permutation of the indices such that the scores are sorted in descending order: _ℎ_ ( _𝑠_ `init`<sup>_𝜋_(1))≥</sup><sup>_ℎ_(</sup><sup>_𝑠_</sup> `init`<sup>_𝜋_(2))≥· · ·≥</sup><sup>_ℎ_(</sup><sup>_𝑠_</sup> `init`<sup>_𝜋_(</sup><sup>_𝑀_)).Then, we initialize the initial solution</sup><sup>_𝑠_0with the top-performing</sup> script, and record the current best score, _i.e._ , _𝑠_ 0 ← _𝑠_ (1) , _ℎ_ `best` ← _ℎ_ ( _𝑠_ 0), where _𝑠_ ( _𝑘_ ) denote the script _𝑠_ `init`<sup>_𝜋_(</sup><sup>_𝑘_)forsimplicity.Finally,wesequentiallyattempttoincorporatetheremainingscripts</sup><sup>_𝑠_(</sup><sup>_𝑘_)for</sup> _𝑘_ = 2 _,_ · · · _, 𝑀_ into _𝑠_ 0. For each _𝑘_ , MLE-STAR creates a candidate merged script by leveraging an agent A `merger` that attempts to integrate _𝑠_ ( _𝑘_ ) into the current _𝑠_ 0. Formally, 



where, A `merger` is guided to introduce a simple average ensemble to merge multiple models. Finally, we merge the models until the validation score _ℎ_ `best` no longer improves (see Appendix B). 

#### **3.2. Refining a code block for solution improvement** 

The iterative refinement phase begins with an initial solution _𝑠_ 0 and proceeds for a predetermined number of _𝑇_ outer loop steps, indexed by _𝑡_ = 0 _,_ 1 _,_ · · · _, 𝑇_ − 1. At each step _𝑡_ , the goal is to improve the current solution _𝑠𝑡_ to obtain _𝑠𝑡_ +1, optimizing for a performance metric _ℎ_ . This process involves two main stages: targeted code block extraction and code block refinement. 

**Targeted code block extraction.** To effectively explore specialized improvement strategies, MLESTAR identifies and targets specific code blocks within the ML pipeline represented by _𝑠𝑡_ . This selection is guided by an ablation study performed by an agent A `abl` . Specifically, the agent A `abl` generates a code _𝑎𝑡_ designed to perform an ablation study on _𝑠𝑡_ . This script creates variations of _𝑠𝑡_ by modifying or disabling specific components. To encourage exploration of different pipeline parts across iterations, A `abl` receives the summaries of previous ablation studies {T `abl`<sup>_𝑖_}</sup><sup>_𝑡_</sup> _𝑖_ =<sup>−</sup> 0<sup>1as input:</sup> 



Then, _𝑎𝑡_ is executed, producing output results _𝑟𝑡_ (see examples in Appendix C). A summarization module A `summarize` processes the script and its results to generate a concise ablation summary T `abl`<sup>_𝑡_:</sup> 



Finally, MLE-STAR selects a critical block which has the most significant impact on performance. To achieve this, an extractor module A `extractor` analyzes the T `abl`<sup>_𝑡_to identify the code block</sup><sup>_𝑐𝑡_within</sup> _𝑠𝑡_ whose modification had the most significant impact on performance. To prioritize refinement of blocks not previously targeted, the set of already refined blocks { _𝑐𝑖_ }<sup>_𝑡_</sup> _𝑖_ =<sup>−</sup> 0<sup>1is provided as context:</sup> 



Here, MLE-STAR also generates the initial plan _𝑝_ 0 for code block refinement at the same time, since T `abl` can provide a good starting point by analyzing the modification of corresponding component. 

**Code block refinement.** Once the targeted code block _𝑐𝑡_ is defined, MLE-STAR explores various refinement strategies to improve the metric _ℎ_ . This involves an inner loop exploring _𝐾_ potential refinement for _𝑐𝑡_ . An agent A `coder` first implements _𝑝_ 0, transforming _𝑐𝑡_ into a refined block _𝑐𝑡_<sup>0,</sup><sup>_i.e._,</sup> _𝑐𝑡_<sup>0= A</sup><sup>`coder`(</sup><sup>_𝑐𝑡, 𝑝_0).A candidate solution</sup><sup>_𝑠_</sup> _𝑡_<sup>0is formed by substituting</sup><sup>_𝑐_</sup> _𝑡_<sup>0into</sup><sup>_𝑠𝑡_:</sup> 



5 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 



<!-- Start of picture text -->
Suggest ensemble plan k Evaluation Ensemble with<br>Task Description + … + the best plan.<br>Dataset Implement plan<br>Update<br>Generate trajectory<br>In parallel Plan 1 → Score 1<br>Feedback to  Plan 2 → Score 2 MLE-STAR Final<br>… Solution<br>Plan k → Score k<br>MLE-STAR<br>… …<br><!-- End of picture text -->







Figure 3 | **Ensembling solutions.** MLE-STAR iteratively proposes effective ensemble strategies based on previous attempts, integrating multiple solutions generated in parallel into a single solution. 

where, `replace` denotes the code replacement operation. Finally, the performance _ℎ_ ( _𝑠𝑡_<sup>0)is evaluated.</sup> 

To discover potentially more effective or novel refinement strategies, MLE-STAR iteratively generates and evaluates further plans. For _𝑘_ = 1 _,_ · · · _, 𝐾_ − 1, a planning agent A `planner` proposes the next plan _𝑝𝑘_ . This agent leverages the previous attempts within the current outer step _𝑡_ as feedback: 



For each plan _𝑝𝑘_ , the coding agent generates the corresponding refined block, _i.e._ , _𝑐𝑡_<sup>_𝑘_= A</sup><sup>`coder`(</sup><sup>_𝑐𝑡, 𝑝𝑘_),</sup> creates the candidate solution _𝑠𝑡_<sup>_𝑘_=</sup><sup>_𝑠𝑡._</sup><sup>`replace`(</sup><sup>_𝑐𝑡, 𝑐_</sup> _𝑡_<sup>_𝑘_),andevaluatesitsperformance</sup><sup>_ℎ_(</sup><sup>_𝑠_</sup> _𝑡_<sup>_𝑘_).After</sup> exploring _𝐾_ refinement strategies (indexed _𝑘_ = 0 _,_ · · · _, 𝐾_ − 1), the best-performing candidate solution is identified: _𝑘_<sup>∗</sup> = arg max _𝑘_ ∈{0 _,_ ··· _,𝐾_ −1} _ℎ_ ( _𝑠𝑡_<sup>_𝑘_).The solution for the next outer step,</sup><sup>_𝑠𝑡_+1, is updated to</sup><sup>_𝑠_</sup> _𝑡_<sup>_𝑘_∗</sup> only if an improvement over _𝑠𝑡_ is found. This iterative process continues until _𝑡_ = _𝑇_ . 

#### **3.3. Further improvement by exploring ensemble strategies** 

To further improve upon the best single solution generated, we introduce a novel ensembling procedure (Figure 3). Standard practice might involve generating multiple candidate solutions and selecting the one with the highest score (Ichihara et al., 2025) according to metric _ℎ_ . However, analogous to model ensembling, we posit that suboptimal solutions might contain complementary strengths, and combining multiple solutions could lead to superior performance compared to relying on any single one. Therefore, we employ the planning capabilities of MLE-STAR to automatically discover effective strategies for ensembling. Specifically, let { _𝑠𝑙_ } _𝑙_<sup>_𝐿_</sup> =1<sup>beasetof</sup><sup>_𝐿_distinctsolutionsobtained</sup> ( _e.g._ , from parallel runs of the process described earlier). Our goal is to find an effective ensemble plan _𝑒_ that merges these solutions, which mirrors the structure of the targeted code block refinement stage. We start with an initial ensemble plan _𝑒_ 0 ( _e.g._ , a simple strategy like averaging the final predictions obtained from the models trained using each solution _𝑠𝑙_ ), proposed by MLE-STAR itself. After the performance _ℎ_ ( _𝑠_ `ens`<sup>0)fortheinitialplan</sup><sup>_𝑒_0iscalculated,forafixednumberofiterations,</sup> _𝑟_ = 1 _,_ · · · _, 𝑅_ , the planning agent A `ens` _ `planner` , specialized in suggesting ensemble plans, proposes subsequent ensemble plans _𝑒𝑟_ . This agent uses the history of previously attempted ensemble plans and their resulting performance as feedback, _i.e._ , _𝑒𝑟_ = A `ens` _ `planner` ({ _𝑠𝑙_ } _𝑙_<sup>_𝐿_</sup> =1<sup>_,_{(</sup><sup>_𝑒𝑗, ℎ_(</sup><sup>_𝑠_</sup> `ens`<sup>_𝑗_))}</sup><sup>_𝑟_</sup> _𝑗_ =<sup>−</sup> 0<sup>1).Each</sup><sup>_𝑒𝑟_</sup> is implemented via A `ensembler` to obtain _𝑠_ `ens`<sup>_𝑟_:</sup> 



Finally, after exploring _𝑅_ ensemble strategies, the ensemble result that achieves the highest performance is selected as the final output, yielding the final ensembled result _𝑠_ `ens`<sup>∗=</sup><sup>_𝑠_</sup> `ens`<sup>_𝑟_∗:</sup><sup>_𝑟_∗=</sup> arg max _𝑟_ ∈{0 _,...,𝑅_ } _ℎ_ ( _𝑠_ `ens`<sup>_𝑟_).ThisprocedureallowsMLE-STARtoautonomouslyexploreandidentify</sup> potentially novel and effective ways to combine multiple complex solutions. 

6 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

#### **3.4. Additional modules for robust MLE agents** 

**Debugging agent.** We detail the design of our debugging agent within MLE-STAR. If the execution of a Python script _𝑠_ triggers an error, resulting in a record T `bug` ( _e.g._ , a traceback), MLE-STAR employs a debugging module A `debugger` to attempt correction. This process iteratively updates the script: 



The debugging step is repeated until either the script executes successfully, or a predefined maximum number of debugging rounds is reached. If the bug cannot be resolved, MLE-STAR proceeds to the next task using the latest version of the script that is known to be executable. 

**Data leakage checker.** We observe that LLM-generated Python scripts might have the risk of introducing data leakage, for example, by improperly accessing information from a test dataset during training dataset preparation (see Figure 6). To address this, we introduce a checker agent, A `leakage` , which analyzes the solution script _𝑠_ prior to its execution. Recognizing that full-script analysis can be inefficient for lengthy code, we adopt a targeted approach. First, we extract the code block _𝑐_ `data` where data preprocessing is done. Second, _𝑐_ `data` is passed to the checker. If A `leakage` detects potential data leakage, it generates a corrected version _𝑐_ `data`<sup>∗:</sup><sup>_𝑐_</sup> `data`<sup>∗= A</sup><sup>`leakage`(</sup><sup>_𝑐_</sup><sup>`data`).Finally, the original script</sup><sup>_𝑠_is</sup> updated by replacing the identified segment with its corrected version: _𝑠_ ← _𝑠._ `replace` ( _𝑐_ `data` _, 𝑐_ `data`<sup>∗).If</sup> no leakage is detected in _𝑐_ `data` by A `leakage` , the script _𝑠_ remains unmodified. All generated solutions are passed through a data leakage checker, A `leakage` , prior to their execution for evaluation. 

**Data usage checker.** We observe that LLM-generated scripts sometimes neglect using provided data sources, focusing solely on simple formats like CSVs (see Figure 7). To ensure the utilization of all relevant provided data, MLE-STAR introduces a data usage checker agent, A `data` . Specifically, before MLE-STAR starts refinement, A `data` checks the initial solution _𝑠_ 0 along with the task description T `task` . If relevant provided data is not adequately used, A `data` revises the initial script as: 



### **4. Experiments** 

In this section, we validate the effectiveness of MLE-STAR using 22 Kaggle competitions from MLEbench Lite (Chan et al., 2025). Our results demonstrate that MLE-STAR significantly outperforms baselines, including those employing various LLMs (Section 4.1). Furthermore, we show that using better models and leveraging our proposed ensemble strategy effectively improves performance (Section 4.2). We also provide the example solutions generated by MLE-STAR, in Appendix D. 

**Common setup.** All experiments are conducted on 22 Kaggle competitions from MLE-bench Lite (Chan et al., 2025) using three random seeds and Gemini-2.0-Flash, unless otherwise specified. Here, we use an agent A `test` , which takes the task description and the final solution as input, and outputs the code that incorporates loading test sample and creating a submission file (see Appendix E for details). MLE-STAR begins by retrieving four model candidates. MLE-STAR refines for four inner loops, while exploring four outer loops. For ensemble, MLE-STAR generates two solutions in parallel, and explore ensemble strategies for five rounds. Following the MLE-bench’s setup, we set a maximum time limit of 24 hours for a fair comparison (see computation analysis in Appendix F). We primarily consider AIDE (Jiang et al., 2025) as our main baseline, given its state-of-the-art performance on MLE-bench. It is important to note that other baselines often limit their generalizability across various task types ( _e.g._ , audio classification, sequence-to-sequence), frequently showcasing results only on simpler modalities like tabular (Hong et al., 2024; Li et al., 2024). For instance, DS-Agent (Guo et al., 2024) requires a manually constructed case bank, and their current GitHub repository lacks cases for audio classification, sequence-to-sequence, image classification, etc. 

7 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

Table 1 | **Main results from MLE-bench Lite.** Each experiment is repeated using three seeds, except for o1-preview (AIDE) and GPT-4o (AIDE), which use 16 and 36 seeds, respectively. All results are taken from the GitHub repository of MLE-bench paper (Chan et al., 2025), except for the model using Gemini. Scores represent the mean and one standard error of the mean. 

|Model|Made<br>Submission<br>(%)|Valid<br>Submission<br>(%)|Above<br>Median<br>(%)|Bronze<br>(%)|Silver<br>(%)|Gold<br>(%)|Any<br>Medal<br>(%)|
|---|---|---|---|---|---|---|---|
|**MLE-STAR (Ours)**||||||||
|**gemini-2.5-pro**<br>gemini-2.0-fash|**100.0**±0.0<br>95.5±2.6|**100.0**±0.0<br>95.5±2.6|**83.3**±4.6<br>63.6±6.0|6.1±3.0<br>**9.1**±3.6|**21.2**±5.1<br>4.5±2.6|**36.4**±6.0<br>30.3±5.7|**63.6**±6.0<br>43.9±6.2|
|**AIDE (Jiang et al., 2025)**||||||||
|gemini-2.0-fash|87.9±4.0|78.8±5.0|39.4±6.0|4.5±2.6|9.1±3.5|12.1±4.0|25.8±5.4|
|o1-preview|99.7±0.3|90.3±1.6|58.2±2.6|4.8±1.1|11.1±1.7|20.7±2.2|36.6±2.6|
|gpt-4o|82.1±1.4|65.7±1.7|29.9±1.6|3.4±0.6|5.8±0.8|9.3±1.0|18.6±1.4|
|llama-3.1-405b-instruct|72.7±5.5|51.5±6.2|18.2±4.7|0.0±0.0|4.5±2.6|6.1±2.9|10.6±3.8|
|claude-3-5-sonnet|81.8±4.7|66.7±5.8|33.3±5.8|3.0±2.1|6.1±2.9|10.6±3.8|19.7±4.9|
|**MLAB (Huang et al., 202**|**4a)**|||||||
|gpt-4o|84.8±4.4|63.6±5.9|7.6±3.3|3.0±2.1|1.5±1.5|1.5±1.5|6.1±2.9|
|**OpenHands (Wang et al., **|**2024)**|||||||
|gpt-4o|81.8±4.7|71.2±5.6|16.7±4.6|3.0±2.1|3.0±2.1|6.1±2.9|12.1±4.0|



Table 2 | Comparison with DS-Agent. 

Table 3 | Performance with Claude-Sonnet-4. 

|Task|Metric|DS-Agent|**MLE-STAR**|Task|Metric|Gemini-2.0-fash|**Sonnet 4**|
|---|---|---|---|---|---|---|---|
|WBY|MAE (↓)|213|**166**|DDD|RMSE (↓)|0.0681|**0.0155**|
|MCC|RMLSE (↓)|0.2964|**0.2911**|DBI|Log Loss (↓)|0.4535|**0.3114**|
|ST|Accuracy (↑)|0.7982|**0.8091**|SAI|Log Loss (↓)|0.2797|**0.2610**|
|ES|AUROC (↑)|0.8727|**0.9101**|WCR|AUROC (↑)|**0.9903**|0.9888|



#### **4.1. Main results** 

**Quantitative results.** As demonstrated in Table 1, MLE-STAR significantly enhances the performance of various baseline models. For instance, when applied to Gemini-2.0-Flash, MLE-STAR improves AIDE’s any medal achieving rates in Kaggle competitions from 25.8% to 43.9%, representing an improvement of over 18 percentage points, and rate of above median from 39.4% to 63.6%. Notably, MLE-STAR with Gemini-2.0-Flash also substantially outperforms AIDE using a powerful reasoning model ( _i.e._ , o1-preview) in terms of achieving gold medals in 10% more tasks. Moreoever, using Gemini-2.5-Pro, MLE-STAR shows a medal achievement rate of over 60%. 

**Comparison to DS-Agent.** While DS-Agent (Guo et al., 2024) shows competitive results on ML tasks, it necessitates human effort to curate its case bank from Kaggle. Consequently, a direct comparison between DS-Agent and AIDE or our method is not feasible, as collecting tasks across diverse modalities, such as audio classification or image denoising, requires additional effort. Nevertheless, we utilize four tabular classification tasks, _i.e._ , wild-blueberry-yield (WBY), media-campaign-cost (MCC), spaceshiptitanic (ST), and enzyme-substrate (ES), the same ones employed during DS-Agent’s development stage (Guo et al., 2024), for a comparison. All experiments are done for 5 seeds following the original setup. As shown in Table 2, MLE-STAR using Gemini-2.0-Flash significantly outperforms DS-Agent even without human efforts. See Appendix G for additional results, including comparison with AutoGluon (Erickson et al., 2020). 

8 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

Table 4 | **Ablation on ensemble strategy.** Experiment results on MLE-bench Lite, repeated three seeds using Gemini-2.0-Flash. Scores represent the mean and one standard error of the mean. 

|Ensemble strategy|Made<br>Submission<br>(%)|Valid<br>Submission<br>(%)|Above<br>Median<br>(%)|Bronze<br>(%)|Silver<br>(%)|Gold<br>(%)|Any<br>Medal<br>(%)|
|---|---|---|---|---|---|---|---|
|**AIDE (Jiang et al., **|**2025)**|||||||
|None|87.9±4.0|78.8±5.0|39.4±6.0|4.5±2.6|9.1±3.5|12.1±4.0|25.8±5.4|
|**MLE-STAR (Ours)**||||||||
|None|**95.5**±2.6|**95.5**±2.6|57.6±6.1|7.6±3.3|4.5±2.6|25.8±5.4|37.9±6.0|
|Best-of-N|**95.5**±2.6|**95.5**±2.6|62.1±6.0|6.1±3.0|7.6±3.3|28.8±5.6|42.4±6.1|
|Average ensemble|**95.5**±2.6|**95.5**±2.6|60.6±6.1|6.1±3.0|**12.1**±4.0|25.8±9.4|**43.9**±6.2|
|**Ours**|**95.5**±2.6|**95.5**±2.6|**63.6**±6.0|**9.1**±3.6|4.5±2.6|**30.3**±5.7|**43.9**±6.2|



#### **4.2. Ablation studies** 

**Performance with reasoning models.** First of all, as shown in Table 1, Gemini-2.5-Pro yields better performance than using Gemini-2.0-Flash. For example, in denoising-dirty-documents cometition, MLE-STAR wigh Gemini-2.0-Flash scored above the median across all three seeds, failing to achieve any medals. However, when using Gemini-2.5-Pro, MLE-STAR achieves two gold medals and one silver medal. These results demonstrate that MLE-STAR is designed to harness the advancements of rapidly improving reasoning-based LLMs. 

In addition, we conducted additional experiments using Claude-Sonnet-4. As shown in Table 3, other models besides Gemini also show promising results, proving compatibility and generalizability in terms of model types. Here, we select four different type of competitions: image-to-image (denoising-dirty-documents; DDD), image classification (dog-breed-identification; DBI), text classification (spooky-author-identification, SAI), and audio classification (the-icml-2013-whale-challengeright-whale-redux; WCR). We run each competition for three seeds. These results indicates that our framework is also compatible and generalizable in terms of LLM type. 

**Effectiveness of proposed ensemble method.** As highlighted in Table 4, MLE-STAR demonstrates a significant performance improvement over the competing baseline, _i.e._ , AIDE, achieving over a 12% higher rate of obtaining any medal _even without_ additional ensemble strategy. Notably, by ensembling multiple solution candidates, our approach yields even greater performance gains, _i.e._ , MLE-STAR consistently improves the success rate for achieving any medal (and specifically gold medals), also surpassing the median human expert’s performance by a larger margin compared to scenarios where this ensembling method is not used. While simpler strategies, such as selecting the solution with the best validation score or averaging final submissions, also offer benefits, MLESTAR shows stronger effectiveness, _e.g._ , leading to a higher number of gold medals. 

### **5. Discussion** 

**Qualitative observations on selected models.** Figure 4 illustrates the model usage of two MLE agents: AIDE and MLE-STAR. AIDE primarily employs ResNet (He et al., 2016) for image classification. However, ResNet, released in 2015, is now considered outdated and can result in suboptimal performance. In contrast, our MLE-STAR primarily utilizes more recent and competitive models like EfficientNet (Tan and Le, 2019) or ViT (Dosovitskiy et al., 2021), leading to the performance gain, winning 37% of the medals, more than AIDE, which wins 26% of the image classification challenges. 

9 



<!-- Start of picture text -->
Baseline MLE-STAR (Ours<br>80<br>60<br>gS 40<br>© 20<br>n<br>> 9 —_m_| a<br>© & & & & <<br>ee SS € & ~<br>we &<br>Model Architecture (Ordered by Release Year)<br><!-- End of picture text -->



<!-- Start of picture text -->
Sid<br>7<br>wyeral 2@))<br><!-- End of picture text -->



<!-- Start of picture text -->
( ee )<br>z<br><!-- End of picture text -->







<!-- Start of picture text -->
—————  SSSSSSSSSSSSSSSSSsses<br>SSS<br><!-- End of picture text -->



<!-- Start of picture text -->
2 22.3<br>20.8<br>&~= 20 17.7<br>= 15<br>g 12.6<br>fo}-2<br>— 10<br><<br>2=<br>3 5<br>$0<br>0<br>StepO Step1 Step2 Step3 Step4<br>Refinement<br><!-- End of picture text -->

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

### **References** 

- T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. Language models are few-shot learners. _Advances in Neural Information Processing Systems_ , 2020. 

- J. S. Chan, N. Chowdhury, O. Jaffe, J. Aung, D. Sherburn, E. Mays, G. Starace, K. Liu, L. Maksin, T. Patwardhan, et al. Mle-bench: Evaluating machine learning agents on machine learning engineering. _International Conference on Learning Representations_ , 2025. 

- T. Chen and C. Guestrin. Xgboost: A scalable tree boosting system. _ACM SIGKDD International Conference on Knowledge Discovery and Data Mining_ , 2016. 

- A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. _International Conference on Learning Representations_ , 2021. 

- T. Elsken, J. H. Metzen, and F. Hutter. Neural architecture search: A survey. _Journal of Machine Learning Research_ , 2019. 

- N. Erickson, J. Mueller, A. Shirkov, H. Zhang, P. Larroy, M. Li, and A. Smola. Autogluon-tabular: Robust and accurate automl for structured data. _arXiv preprint arXiv:2003.06505_ , 2020. 

- L. Fan, F. Zhang, H. Fan, and C. Zhang. Brief review of image denoising techniques. _Visual computing for industry, biomedicine, and art_ , 2019. 

- W. Fan, E. Zhong, J. Peng, O. Verscheure, K. Zhang, J. Ren, R. Yan, and Q. Yang. Generalized and heuristic-free feature construction for improved accuracy. _SIAM International Conference on Data Mining_ , 2010. 

- M. Feurer, K. Eggensperger, S. Falkner, M. Lindauer, and F. Hutter. Auto-sklearn 2.0: Hands-free automl via meta-learning. _Journal of Machine Learning Research_ , 2022. 

- S. Guo, C. Deng, Y. Wen, H. Chen, Y. Chang, and J. Wang. DS-agent: Automated data science by empowering large language models with case-based reasoning. _International Conference on Machine Learning_ , 2024. 

- K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. _IEEE Conference on Computer Vision and Pattern Recognition_ , 2016. 

- N. Hollmann, S. Müller, and F. Hutter. Large language models for automated data science: Introducing caafe for context-aware automated feature engineering. _Advances in Neural Information Processing Systems_ , 2023. 

- N. Hollmann, S. Müller, L. Purucker, A. Krishnakumar, M. Körfer, S. B. Hoo, R. T. Schirrmeister, and F. Hutter. Accurate predictions on small data with a tabular foundation model. _Nature_ , 2025. 

- D. Holzmüller, L. Grinsztajn, and I. Steinwart. Better by default: Strong pre-tuned mlps and boosted trees on tabular data. _Advances in Neural Information Processing Systems_ , 2024. 

- S. Hong, Y. Lin, B. Liu, B. Liu, B. Wu, C. Zhang, C. Wei, D. Li, J. Chen, J. Zhang, et al. Data interpreter: An llm agent for data science. _arXiv preprint arXiv:2402.18679_ , 2024. 

- F. Horn, R. Pack, and M. Rieger. The autofeat python library for automated feature engineering and selection. _Joint European Conference on Machine Learning and Knowledge Discovery in Databases_ , 2019. 

12 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

- X. Hu, Z. Zhao, S. Wei, Z. Chai, Q. Ma, G. Wang, X. Wang, J. Su, J. Xu, M. Zhu, et al. Infiagent-dabench: Evaluating agents on data analysis tasks. _arXiv preprint arXiv:2401.05507_ , 2024. 

- Q. Huang, J. Vora, P. Liang, and J. Leskovec. Mlagentbench: Evaluating language agents on machine learning experimentation. _International Conference on Machine Learning_ , 2024a. 

- Y. Huang, J. Luo, Y. Yu, Y. Zhang, F. Lei, Y. Wei, S. He, L. Huang, X. Liu, J. Zhao, et al. Dacode: Agent data science code generation benchmark for large language models. _arXiv preprint arXiv:2410.07331_ , 2024b. 

- Y. Ichihara, Y. Jinnai, T. Morimura, K. Abe, K. Ariu, M. Sakamoto, and E. Uchibe. Evaluation of best-of-n sampling strategies for language model alignment. _Transactions on Machine Learning Research_ , 2025. 

- N. Jain, K. Han, A. Gu, W.-D. Li, F. Yan, T. Zhang, S. Wang, A. Solar-Lezama, K. Sen, and I. Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. _International Conference on Learning Representations_ , 2025. 

- Z. Jiang, D. Schmidt, D. Srikanth, D. Xu, I. Kaplan, D. Jacenko, and Y. Wu. Aide: Ai-driven exploration in the space of code. _arXiv preprint arXiv:2502.13138_ , 2025. 

- C. E. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press, and K. Narasimhan. Swe-bench: Can language models resolve real-world github issues? _International Conference on Learning Representations_ , 2024. 

- H. Jin, Q. Song, and X. Hu. Auto-keras: An efficient neural architecture search system. _ACM SIGKDD International Conference on Knowledge Discovery and Data Mining_ , 2019. 

- L. Jing, Z. Huang, X. Wang, W. Yao, W. Yu, K. Ma, H. Zhang, X. Du, and D. Yu. Dsbench: How far are data science agents to becoming data science experts? _International Conference on Learning Representations_ , 2025. 

- J. M. Kanter and K. Veeramachaneni. Deep feature synthesis: Towards automating data science endeavors. _IEEE International Conference on Data Science and Advanced Analytics_ , 2015. 

- J. L. Kolodner. An introduction to case-based reasoning. _Artificial intelligence review_ , 1992. 

- L. Kotthoff, C. Thornton, H. H. Hoos, F. Hutter, and K. Leyton-Brown. Auto-weka 2.0: Automatic model selection and hyperparameter optimization in weka. _Journal of Machine Learning Research_ , 2017. 

- E. LeDell and S. Poirier. H2O AutoML: Scalable automatic machine learning. _ICML Workshop on AutoML_ , 2020. 

- L. Li, H. Wang, L. Zha, Q. Huang, S. Wu, G. Chen, and J. Zhao. Learning a data-driven policy network for pre-training automated feature engineering. _International Conference on Learning Representations_ , 2023. 

- Y. Li, D. Choi, J. Chung, N. Kushman, J. Schrittwieser, R. Leblond, T. Eccles, J. Keeling, F. Gimeno, A. Dal Lago, et al. Competition-level code generation with alphacode. _Science_ , 2022. 

- Z. Li, Q. Zang, D. Ma, J. Guo, T. Zheng, M. Liu, X. Niu, Y. Wang, J. Yang, J. Liu, et al. Autokaggle: A multi-agent framework for autonomous data science competitions. _arXiv preprint arXiv:2410.20424_ , 2024. 

13 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

- J. Nam, K. Kim, S. Oh, J. Tack, J. Kim, and J. Shin. Optimized feature generation for tabular data via llms with decision tree reasoning. _Advances in Neural Information Processing Systems_ , 2024. 

- R. S. Olson and J. H. Moore. Tpot: A tree-based pipeline optimization tool for automating machine learning. _ICML Workshop on AutoML_ , 2016. 

- F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, et al. Scikit-learn: Machine learning in python. _Journal of Machine Learning Research_ , 2011. 

- H. Pham, M. Guan, B. Zoph, Q. Le, and J. Dean. Efficient neural architecture search via parameters sharing. _International Conference on Machine Learning_ , 2018. 

- L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, and A. Gulin. Catboost: unbiased boosting with categorical features. _Advances in Neural Information Processing Systems_ , 2018. 

- E. Real, A. Aggarwal, Y. Huang, and Q. V. Le. Regularized evolution for image classifier architecture search. _AAAI Conference on Artificial Intelligence_ , 2019. 

- S. Schmidgall, Y. Su, Z. Wang, X. Sun, J. Wu, X. Yu, J. Liu, Z. Liu, and E. Barsoum. Agent laboratory: Using llm agents as research assistants. _arXiv preprint arXiv:2501.04227_ , 2025. 

- Y. Shen, K. Song, X. Tan, D. Li, W. Lu, and Y. Zhuang. Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face. _Advances in Neural Information Processing Systems_ , 2023. 

- M. Tan and Q. Le. Efficientnet: Rethinking model scaling for convolutional neural networks. _International Conference on Machine Learning_ , 2019. 

- G. Team, P. Georgiev, V. I. Lei, R. Burnell, L. Bai, A. Gulati, G. Tanzer, D. Vincent, Z. Pan, S. Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. _arXiv preprint arXiv:2403.05530_ , 2024. 

- H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, et al. Llama: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_ , 2023. 

- G. Wang, Y. Xie, Y. Jiang, A. Mandlekar, C. Xiao, Y. Zhu, L. Fan, and A. Anandkumar. Voyager: An open-ended embodied agent with large language models. _arXiv preprint arXiv: Arxiv-2305.16291_ , 2023. 

- X. Wang, B. Li, Y. Song, F. F. Xu, X. Tang, M. Zhuge, J. Pan, Y. Song, B. Li, J. Singh, et al. Openhands: An open platform for ai software developers as generalist agents. _International Conference on Learning Representations_ , 2024. 

- I. Watson and F. Marir. Case-based reasoning: A review. _The knowledge engineering review_ , 1994. 

- S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao. React: Synergizing reasoning and acting in language models. _International Conference on Learning Representations_ , 2023. 

- Z. You, Y. Zhang, D. Xu, Y. Lou, Y. Yan, W. Wang, H. Zhang, and Y. Huang. Datawiseagent: A notebookcentric llm agent framework for automated data science. _arXiv preprint arXiv:2503.07044_ , 2025. 

- T. Zhang, Z. A. Zhang, Z. Fan, H. Luo, F. Liu, Q. Liu, W. Cao, and L. Jian. Openfe: Automated feature generation with expert-level performance. _International Conference on Machine Learning_ , 2023. 

- B. Zoph and Q. V. Le. Neural architecture search with reinforcement learning. _International Conference on Learning Representations_ , 2017. 

14 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

## **Appendix** 

### **A. Prompts for MLE-STAR** 

#### **A.1. Retriever agent** 

**# Competition {task description} # Your task - List {** **_M_ } recent effective models and their example codes to win the above competition. # Requirement - The example code should be concise and simple. - You must provide an example code, i.e., do not just mention GitHubs or papers. Use this JSON schema: Model = {'model_name': str, 'example_code': str} Return: list[Model]** 

Figure 9 | Prompt used for retrieving task-specific models using web search. 

MLE-STAR starts by generating an initial solution. Here, we propose using web search as a tool for MLE-STAR first to retrieve _𝑀_ state-of-the-art models for the given task. Specifically, MLESTAR leverages a retriever agent A `retriever` with the above prompt (Figure 9). A `retriever` takes task description T `task` as input and retrieves _𝑀_ pairs of {T `model` _,_ T `code` }. Here, we guide MLE-STAR to generate the retrieved result as structured output ( _i.e._ , JSON). After we obtain JSON file, we parse them into separate model cards.<sup>2</sup> 

> 2See `example_intermediate_outputs/retriever_output.txt` in `https://github.com/jaehyun513/ MLE-STAR` . 

15 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

#### **A.2. Candidate evaluation agent** 

- **# Introduction - You are a Kaggle grandmaster attending a competition. - We will now provide a task description and a model description. - You need to implement your Python solution using the provided model. # Task description {task description} # Model description ## Model name {model description} ## Example Python code {example code} # Your task - Implement the solution in Python. - You must use the model as described in the model description. - This first solution design should be relatively simple, without ensembling or hyper-parameter optimization. - Propose an evaluation metric that is reasonable for this task. - All the provided data is already prepared and available in the `./input` directory. There is no need to unzip any files. - Do not include other models that are not directly related to the model described. - Use PyTorch rather than TensorFlow. Use CUDA if you need. All the necessary libraries are installed. - The code should implement the proposed solution and print the value of the evaluation metric computed on a hold-out validation set. - Only use the provided train data in the `./input` directory. Do not load test data. - If there are more than 30,000 training samples, you must subsample to 30,000 for a faster run. # Required - There should be no additional headings or text in your response. - Print out or return a final performance metric in your answer in a clear format with the exact words: 'Final Validation Performance: {final_validation_score}'. - The code should be a single-file Python program that is self-contained and can be executed as-is. - Your response should only contain a single code block. - Do not use exit() function in the Python code. - Do not use try: and except: or if else to ignore unintended behavior.** 

#### Figure 10 | Prompt used for evaluating retrieved models. 

MLE-STAR uses candidate evaluation agent A `init` to evaluate the performance of the retrieved model. As shown in Figure 10, by taking task description (T `task` ), model description (T `model` ), and corresponding code example (T `code` ), A `init` generates a Python script.<sup>3</sup> The Python script for the retrieved model evaluation is guided to be relatively simple, and to contain the evaluation result computed on a hold-out validation set. In addition, if there are too many training samples, A `init` uses the subset of training sample for faster execution. 

> 3See `example_intermediate_outputs/candidate_evaluation.py` in `https://github.com/jaehyun513/ MLE-STAR` for an example. 

16 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

#### **A.3. Merging agent** 



<!-- Start of picture text -->
# Introduction<br>- You are a Kaggle grandmaster attending a competition.<br>- We will now provide a base solution and an additional reference solution.<br>- You need to implement your Python solution by integrating reference solution to the base<br>solution.<br># Base solution<br>{base code}<br># Reference solution<br>{reference code}<br># Your task<br>- Implement the solution in Python.<br>- You have to integrate the reference solution to the base solution.<br>- Your code base should be the base solution.<br>- Try to train additional model of the reference solution.<br>- When integrating, try to keep code with similar functionality in the same place (e.g.,<br>all preprocessing should be done and then all training).<br>- When integrating, ensemble the models.<br>- The solution design should be relatively simple.<br>- The code should implement the proposed solution and print the value of the evaluation<br>metric computed on a hold-out validation set.<br>- Only use the provided train data in the `./input` directory.<br>- If there are more than 30,000 training samples, you must subsample to 30,000 for a faster<br>run.<br># Required<br>- There should be no additional headings or text in your response.<br>- Print out or return a final performance metric in your answer in a clear format with the<br>exact words: 'Final Validation Performance: {final_validation_score}'.<br>- The code should be a single-file Python program that is self-contained and can be<br>executed as-is.<br>- Your response should only contain a single code block.<br>- Do not use exit() function in the Python code.<br>- Do not use try: and except: or if else to ignore unintended behavior<br><!-- End of picture text -->

Figure 11 | Prompt used for merging the candidate models for generating initial solution. 

MLE-STAR leverages an agent A `merger` to merge the retrieved models into a consolidated initial solution. As shown in Figure 11, this process is done sequentially, where the prompt guides the agent to integrate the reference code ( _i.e._ , the best candidate model code among the models that are not merged yet) into the base code ( _i.e._ , the current candidate merged script). The output of A `merger` is a Python script<sup>4</sup> , which will be the next candidate merged script. See Appendix B for the sequential procedure of MLE-STAR when generating the initial solution. 

> 4See `example_intermediate_outputs/merged_candidate.py` in `https://github.com/jaehyun513/ MLE-STAR` for an example. 

17 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

#### **A.4. Ablation study agent** 

**# Introduction** 

**- You are a Kaggle grandmaster attending a competition. - In order to win this competition, you need to perform an ablation study on the current Python solution to know which parts of the code contribute the most to the overall performance.** 

**- We will now provide a current Python solution. - We will also provide the summaries of previous ablation studies.** 

**# Python solution {solution script} ## Previous ablation study result {0} {previous_ablations[0]} ## Previous ablation study result {1} {previous_ablations[1]} ...** 

**## Previous ablation study result {t-1} {previous_ablations[t-1]}** 

**# Instructions - You need you to generate a simple Python code that performs an ablation study on the train.py script.** 

- **The generated code should create variations by modifying or disabling parts (2-3 parts) of the training process.** 

- **Your ablation study should concentrate on the other parts that have not been previously considered.** 

- **For each ablation, print out how the modification affects the model's performance.** 

##### **# Response format** 

**- There should be no additional headings or text in your response.** 

- **The Python code for the ablation study should not load test data. It should only focus on training and evaluating the model on the validation set.** 

**- The code should include a printing statement that shows the performance of each ablation. - The code should consequently print out what part of the code contributes the most to the overall performance.** 

Figure 12 | Prompt used for generating a Python script for ablation studies. 

To effectively explore specialized improvement strategies, MLE-STAR identifies and targets specific code blocks. This code block selection is guided by an ablation study performed by an agent A `abl` . As shown in Figure 12, A `abl` generates a Python code designed to perform an ablation study on current solution. The prompt guides the agent to modify or disable specific component.<sup>5</sup> Moreover, to encourage exploration of different pipeline parts, the summaries of previous ablation studies are also used as valuable feedback. See Appendix C for the example output of the agent A `abl` . 

> 5We provide an example generated code for ablation study (which is generated by A `abl` ) in `https://github.com/ jaehyun513/MLE-STAR` (see `example_intermediate_outputs/ablation.py` ). 

18 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

#### **A.5. Ablation study summarization agent** 

**# Your code for ablation study was: {code for ablation study}** 

- **# Ablation study results after running the above code:** 

**{raw result}** 

**# Your task - Summarize the result of ablation study based on the code and printed output.** 

#### Figure 13 | Prompt used for summarizing the result of the ablation study. 

After executing the code for an ablation study, denoted as _𝑎𝑡_ , the output result _𝑟𝑡_ is produced. Since _𝑟𝑡_ often contains content unrelated to the ablation (for example, printing the loss value across training epochs), a summarization module A `summarize` is utilized with the prompt mentioned above (Figure 13). This module takes _𝑎𝑡_ and _𝑟𝑡_ as input to summarize and parse the ablation study results. Here, _𝑎𝑡_ is also used because it provides information about the modification. See Appendix C for the examples of _𝑟𝑡_ and the summarization result. 

19 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

#### **A.6. Extractor** 

|**# Introduction**<br>**- You are a Kaggle grandmaster attending a competition.**<br>**- In order to win this competition, you need to extract a code block from the current**<br>**Python solution and improve the extracted block for better performance.**<br>**- Your suggestion should be based on the ablation study results of the current Python**<br>**solution.**<br>**- We will now provide the current Python solution and the ablation study results.**<br>**- We also provide code blocks which you have tried to improve previously.**<br>**# Python solution**<br>**{solution script}**<br>**# Ablation study results**<br>**{summary of ablation study}**<br>**## Code block{0}**<br>**{prev_code_blocks[0]}**<br>**## Code block{1}**<br>**{prev_code_blocks[1]}**<br>**...**<br>**## Code block{t-1}**<br>**{prev_code_blocks[t-1]}**<br>**# Your task**<br>**- Given the ablation study results, suggest an effective next plan to improve the above**<br>**Python script.**<br>**- The plan should be a brief outline/sketch of your proposed solution in natural language**<br>**(3-5 sentences).**<br>**- Please avoid plan which can make the solution's running time too long (e.g., searching**<br>**hyperparameters in a very large search space).**<br>**- Try to improve the other part which was not considered before.**<br>**- Also extract the code block from the above Python script that need to be improved**<br>**according to the proposed plan. You should try to extract the code block which was not**<br>**improved before.**<br>**# Response format**<br>**- Your response should be a brief outline/sketch of your proposed solution in natural**<br>**language (3-5 sentences) and a single markdown code block which is the code block that need**<br>**to be improved.**<br>**- The code block can be long but should be exactly extracted from the Python script**<br>**provided above.**<br>**Use this JSON schema:**|
|---|
|**Refine_Plan = {'code_block': str, 'plan': str}**<br>**Return: list[Refine_Plan]**|



Figure 14 | Prompt used for extracting the code block that has the most significant impact. 

MLE-STAR uses an extractor module A `extractor` to analyze the T `abl` and then identify the code block _𝑐𝑡_ . As shown in Figure 14, A `extractor` takes the summary of the ablation study, current solution code, and the previously refined code blocks as input, and is guided to output the code block which has the most significant impact on performance. Here, the initial plan for refining the extracted code block is also generated.<sup>6</sup> 

> 6See `example_intermediate_outputs/code_block.txt` in `https://github.com/jaehyun513/MLE-STAR` for an example of extracted code block. 

20 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

#### **A.7. Coder** 

**# Introduction - You are a Kaggle grandmaster attending a competition. - In order to win this competition, you need refine the code block for better performance based on the improvement plan. - We will now provide the code block and the improvement plan. # Code block {code_block} # Improvement plan {plan} # Your task - Implement the improvement plan on the above code block. But do not remove subsampling if exists. - The code block should be improved according to the proposed plan. - Note that all the variable including actual data is defined earlier (since you are just seeing a code block), therefore do not introduce dummy variables. # Response format - Your response should be a single markdown code block (wrapped in ```) which is the improved code block. - There should be no additional headings or text in your response.** 

Figure 15 | Prompt used for implementing refinement plan on the extracted code block. 

The implementation of code block refinement is done by A `coder` , which takes the extracted code block and the refinement plan as input, and outputs the refined code block.<sup>7</sup> 

> 7We provide an example of the target code block, proposed plan, and the output of refined code block by A `coder` in `https://github.com/jaehyun513/MLE-STAR` (see `example` _ `intermeidate` _ `outputs` / `coder` _ `outputs` / directory). 

21 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

#### **A.8. Planner** 

**# Introduction - You are a Kaggle grandmaster attending a competition. - In order to win this competition, you have to improve the code block for better performance. - We will provide the code block you are improving and the improvement plans you have tried. # Code block {code block} # Improvement plans you have tried ## Plan: {plans[0]} ## Score: {scores[0]} ## Plan: {plans[1]} ## Score: {scores[1]} ... ## Plan: {plans[k-1]} ## Score: {scores[k-1]} # Your task - Suggest a better plan to improve the above code block. - The suggested plan must be novel and effective. - Please avoid plans which can make the solution's running time too long (e.g., searching hyperparameters in a very large search space). - The suggested plan should be differ from the previous plans you have tried and should receive a higher score.** 

**# Response format - Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences). - There should be no additional headings or text in your response.** 

Figure 16 | Prompt used for generating the next refinement plan which targets the extracted code block. 

To discover potentially more effective or novel refinement strategies (targeting the extracted code block), MLE-STAR iteratively generates further plans through a planning agent A `planner` . As shown in Figure 16, A `planner` takes the extracted code block and the previous attempts as input and proposes the next plan. These are examples of proposed plans: 

22 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

- `f'''Since feature engineering had the biggest impact, I will focus on improving the cabin` _↩_ → `feature extraction. Instead of simply splitting the Cabin string, I will create dummy` _↩_ → `variables for each unique Deck and Side. Also, the Cabin_num will be kept as` _↩_ → `numerical, imputing missing values using a median strategy to handle potential` _↩_ → `outliers. This approach should provide more granular information to the models.'''` 

`f'''Instead of one-hot encoding 'Deck' and 'Side' directly, I will explore interaction` _↩_ → `features between 'Deck', 'Side', and potentially 'Cabin_num'. Specifically, I'll` _↩_ → `create combined features like 'Deck_Side' and 'Deck_Cabin_num' to capture potential` _↩_ → `dependencies. Furthermore, I will impute missing 'Cabin_num' values using a more` _↩_ → `sophisticated method like k-NN imputation, considering other features like 'Deck',` _↩_ → `'Side', and 'RoomService' to improve imputation accuracy. This should capture more` _↩_ → `complex relationships within the cabin data and lead to better model performance.'''` 

`f'''I propose a plan that focuses on a more nuanced approach to 'Cabin_num' and` _↩_ → `interaction terms. First, I'll bin 'Cabin_num' into ordinal categories (e.g., low,` _↩_ → `medium, high) based on quantiles, as the absolute number might not be as important as` _↩_ → `its relative position. Then, I'll create interaction features between the binned` _↩_ → `'Cabin_num', 'Deck', and 'Side' using one-hot encoding. This will allow the model to` _↩_ → `learn specific combinations of cabin location and number range that might be` _↩_ → `predictive. Finally, I will use a simple imputer for the missing values in` _↩_ → `'Cabin_num' before binning.'''` 

23 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

#### **A.9. Ensemble strategy planner** 

**# Introduction - You are a Kaggle grandmaster attending a competition. - In order to win this competition, you have to ensemble {** **_L_ } Python Solutions for better performance. - We will provide the Python Solutions and the ensemble plans you have tried. # 1st Python Solution {solution1} # 2nd Python Solution {solution2} ... # {** **_L_ }th Python Solution {solution** **_L_ } # Ensemble plans you have tried ## Plan: {plans[0]} ## Score: {scores[0]} ## Plan: {plans[1]} ## Score: {scores[1]} ... ## Plan: {plans[r-1]} ## Score: {scores[r-1]} # Your task - Suggest a better plan to ensemble the {** **_L_ } solutions. You should concentrate how to merge, not the other parts like hyperparameters.** 

**- The suggested plan must be easy to implement, novel, and effective. - The suggested plan should be differ from the previous plans you have tried and should receive a higher (or lower) score.** 

**# Response format** 

**- Your response should be an outline/sketch of your proposed solution in natural language.** 

**- There should be no additional headings or text in your response.** 

**- Plan should not modify the original solutions too much since execution error can occur.** 

Figure 17 | Prompt used for generating the next ensemble plan. 

on the history of previously attempted ensemble plans and their resulting performance as feedback.As shown in Figure 17, similar to A `planner` , A `ens` _ `planner` proposes an effective ensemble plan based These are examples of attempted ensemble plans. 

24 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

- `f'''Averaging the predicted probabilities from both models is a straightforward and` _↩_ → `effective ensembling technique. First, modify the AutoGluon solution to output` _↩_ → `probabilities instead of hard predictions using `predictor.predict_proba(test_data)`.` _↩_ → `Then, obtain the predicted probabilities from the LightGBM model using` _↩_ → ``lgbm_classifier.predict_proba(X_test_processed)`. Average these probabilities for` _↩_ → `each class. Finally, generate the final predictions by thresholding the averaged` _↩_ → `probability of the 'Transported' class at 0.5. Create the submission file based on` _↩_ → `these averaged and thresholded predictions.'''` 

```
f'''Here'sanensemblingplanleveragingstackingwithasimplemeta-learner:
```

`1. **Generate Predictions:** Use both the AutoGluon model and the LightGBM model to` _↩_ → `generate predictions on the original training data (train.csv). This is crucial for` _↩_ → `training the meta-learner. For AutoGluon, use `predictor.predict_proba(train_data)`` _↩_ → `and extract the probabilities for the 'Transported' class. For LightGBM, preprocess` _↩_ → `the training data using the same pipeline as the test data and get probabilities with` _↩_ → ``lgbm_classifier.predict_proba(X_processed)` and again, extract the probabilities for` _↩_ → `the 'Transported' class.` 

`2. **Create Meta-Features:** Combine the predicted probabilities from AutoGluon and` _↩_ → `LightGBM for the training data into a new dataframe. This dataframe will have two` _↩_ → `columns: 'AutoGluon_Prob' and 'LGBM_Prob', and the 'Transported' column from the` _↩_ → `original training data as the target variable for the meta-learner.` 

`3. **Train Meta-Learner:** Use a simple model like Logistic Regression as the` _↩_ → `meta-learner. Train this Logistic Regression model using the meta-features` _↩_ → `(AutoGluon_Prob, LGBM_Prob) to predict the 'Transported' column. This step aims to` _↩_ → `learn how to best combine the predictions of the base models.` 

`4. **Generate Test Predictions:** Get the predicted probabilities from AutoGluon and` _↩_ → `LightGBM on the test set, as in the averaging approach.` 

`5. **Create Meta-Features for Test Data:** Create a dataframe for the test data, with` _↩_ → `the same structure as the training meta-features (AutoGluon_Prob, LGBM_Prob) from the` _↩_ → `test set.` 

`6. **Meta-Learner Prediction:** Use the trained Logistic Regression model to predict the` _↩_ → `final 'Transported' probabilities on the test meta-features.` 

`7. **Threshold and Submit:** Threshold the predicted probabilities from the meta-learner` _↩_ → `at 0.5 to get the final predictions (True/False) and create the submission file.'''` 

25 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

- `f'''Here's an ensembling plan that focuses on weighted averaging with optimized weights` _↩_ → `determined by a simple grid search on a validation set:` 

- `1. **Validation Split:** Split the original training data into two parts: a training set` _↩_ → `(e.g., 80% of the data) and a validation set (e.g., 20% of the data). Crucially,` _↩_ → `perform the preprocessing steps (OneHotEncoding, Scaling, etc.) separately on the` _↩_ → `training and validation sets to avoid data leakage.` 

- `2. **Generate Validation Predictions:** Use both the AutoGluon model and the LightGBM` _↩_ → `model to generate predictions on the validation set. For AutoGluon, obtain` _↩_ → `probabilities using `predictor.predict_proba(validation_data)`. For LightGBM,` _↩_ → `preprocess the validation data using the same pipeline trained on the training split` _↩_ → `and get probabilities using `lgbm_classifier.predict_proba(X_validation_processed)`.` 

- `3. **Grid Search for Optimal Weights:** Define a grid of weights for AutoGluon and` _↩_ → `LightGBM. For instance, iterate through weights from 0.0 to 1.0 in increments of 0.1` _↩_ → `for AutoGluon, with the LightGBM weight being (1 - AutoGluon weight). For each weight` _↩_ → `combination: * Calculate the weighted average of the predicted probabilities from AutoGluon and` _↩_ → `LightGBM on the validation set.` 

- `* Threshold the averaged probabilities at 0.5 to obtain binary predictions. * Calculate the accuracy of these predictions against the true labels in the` _↩_ → `validation set.` 

- `4. **Select Best Weights:** Choose the weight combination that yields the highest` _↩_ → `accuracy on the validation set.` 

- `5. **Generate Test Predictions:** Obtain the predicted probabilities from AutoGluon and` _↩_ → `LightGBM on the test set, as before.` 

- `6. **Weighted Averaging on Test Set:** Use the optimal weights determined in step 4 to` _↩_ → `calculate the weighted average of the predicted probabilities from AutoGluon and` _↩_ → `LightGBM on the test set.` 

- `7. **Threshold and Submit:** Threshold the weighted average probabilities at 0.5 to` _↩_ → `obtain the final predictions and create the submission file.` 

- `This plan is easy to implement, avoids complex meta-learners that can overfit, and` _↩_ → `focuses on finding the best combination of the two models based on a validation set.` _↩_ → `It adapts to the strengths of each model by giving them different weights.'''` 

26 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

#### **A.10. Ensembler** 

- **# Introduction - You are a Kaggle grandmaster attending a competition. - In order to win this competition, you need to ensemble {** **_L_ } Python Solutions for better performance based on the ensemble plan. - We will now provide the Python Solutions and the ensemble plan.** 

- **# 1st Python Solution {solution1} # 2nd Python Solution {solution2} ... # {** **_L_ }th Python Solution {solution** **_L_ } # Ensemble Plan {plan} # Your task - Implement the ensemble plan with the provided solutions. - Unless mentioned in the ensemble plan, do not modify the original Python Solutions too much." - All the provided data (except previous submissions; do not load submissions) is already prepared and available in the `.\input` directory. There is no need to unzip any files. - The code should implement the proposed solution and print the value of the evaluation metric computed on a hold-out validation set.** 

##### **# Response format required** 

- **Your response should be a single markdown code block (wrapped in ```) which is the ensemble of {** **_L_ } Python Solutions. - There should be no additional headings or text in your response. - Do not subsample or introduce dummy variables. You have to provide full new Python Solution using the {** **_L_ } provided solutions. - Do not forget the `./final/submission.csv` file. - Print out or return a final performance metric in your answer in a clear format with the exact words: 'Final Validation Performance: {final_validation_score}'. - The code should be a single-file Python program that is self-contained and can be executed as-is.** 

Figure 18 | Prompt used for implementing ensemble plan on the solutions generated by MLE-STAR in parallel. 

The proposed ensemble plan is implemented by A `ensembler` . This agent takes the two final solutions which is generated in parallel by MLE-STAR, and the ensemble plan as input, and outputs the Python script, _i.e._ , the merged code solution (see Appendix C for examples since the final solution is selected among the merged code solution). 

27 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

#### **A.11. Debugging agent** 

**# Code with an error: {code} # Error: {bug} # Your task - Please revise the code to fix the error. - Do not remove subsampling if exists. - Provide the improved, self-contained Python script again. - There should be no additional headings or text in your response. - All the provided input data is stored in "./input" directory. - Remember to print a line in the code with 'Final Validation Performance: {final_validation_score}' so we can parse performance. - The code should be a single-file python program that is self-contained and can be executed as-is. - Your response should only contain a single code block. - Do not use exit() function in the refined Python code.** 

#### Figure 19 | Prompt used for debugging. 

If the execution of a Python script triggers an error, MLE-STAR employs a debugging module A `debugger` to attempt correction using the above prompt (Figure 19). 

28 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

#### **A.12. Data leakage checker** 

**# Python code {code} # Your task - Extract the code block where the validation and test samples are preprocessed using training samples.** 

**- Check that the model is trained with only training samples. - Check that before printing the final validation score, the model is not trained the validation samples. - Also check whether the validation and test samples are preprocessed correctly, preventing information from the validation or test samples from influencing the training process (i.e., preventing data leakage).** 

**# Requirement - Extract a code block and also check the data leakage. - The code block should be an exact subset of the above Python code. - Your response for a code block should be a single markdown code block. - If data leakage is present on validation and test samples, answer 'Yes Data Leakage'. - If data leakage is not present on validation and test samples, answer 'No Data Leakage'. Use this JSON schema: Answer = {'leakage_status': str, 'code_block': str} Return: list[Answer]** 

Figure 20 | Prompt used for extract the code block whether data preprocessing is done. 

**# Python code {code} # Your task - In the above Python code, the validation and test samples are influencing the training process, i.e., not correctly preprocessed.** 

- **Ensure that the model is trained with only training samples.** 

**- Ensure that before printing the final validation score, the model is not trained on the validation samples.** 

- **Refine the code to prevent such data leakage problem. # Requirement - Your response should be a single markdown code block.** 

- **Note that all the variables are defined earlier. Just modify it with the above code.** 

Figure 21 | Prompt used for correcting the code block with a risk of data leakage. 

To mitigate the risk of introducing data leakage, MLE-STAR first extract the code block where preprocessing is done. This is achieved by using the above prompt in Figure 20, which takes the current solution script as input, and then generates (1) the code block and (2) whether the extracted code block has a risk of data leakage. If leakage is detected, the code block is corrected with the prompt in Figure 21, and MLE-STAR replaces the original code block to the corrected version. 

29 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

#### **A.13. Data usage checker** 



<!-- Start of picture text -->
I have provided Python code for a machine learning task (attached below):<br># Solution Code<br>{initial solution}<br>Does above solution code uses all the information provided for training? Here is task<br>description and some guide to handle:<br># Task description<br>{task description}<br># Your task<br>- If the above solution code does not use the information provided, try to incorporate all.<br>Do not bypass using try-except.<br>- DO NOT USE TRY and EXCEPT; just occur error so we can debug it!<br>- See the task description carefully, to know how to extract unused information<br>effectively.<br>- When improving the solution code by incorporating unused information, DO NOT FORGET to<br>print out 'Final Validation Performance: {final_validation_score}' as in original solution<br>code.<br># Response format:<br>Option 1: If the code did not use all the provided information, your response should be a<br>single markdown code block (wrapped in ```) which is the improved code block. There should<br>be no additional headings or text in your response<br>Option 2: If the code used all the provided information, simply state that "All the<br>provided information is used.<br><!-- End of picture text -->

Figure 22 | Prompt used for data usage checker. 

To ensure the utilization of all relevant provided data, MLE-STAR utilizes a data usage checker agent A `data` . This agent checks the initial solution with the task description, and revise the initial script using the prompt in Figure 22. 

30 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

### **B. Algorithms** 

#### **B.1. Algorithm for generating an initial solution** 

**Algorithm 1** Generating an initial solution 

1: **Input** : task description T `task` , datasets D, score function _ℎ_ , number of retrieved models _𝑀_ , 2: {T `model`<sup>_𝑖,_T</sup> `code`<sup>_𝑖_}</sup> _𝑖_<sup>_𝑀_</sup> =1<sup>= A</sup><sup>`retriever`(T</sup><sup>`task`)</sup> 3: **for** _𝑖_ = 1 to _𝑀_ **do** 4: _𝑠_ `init`<sup>_𝑖_= A</sup><sup>`init`(T</sup><sup>`task`</sup><sup>_,_T</sup> `model`<sup>_𝑖,_T</sup> `code`<sup>_𝑖_)</sup> 5: Evaluate _ℎ_ ( _𝑠_ `init`<sup>_𝑖_)using D</sup> 6: **end for** 7: _𝑠_ 0 ← _𝑠_ `init`<sup>_𝜋_(1)</sup> 8: _ℎ_ `best` ← _ℎ_ ( _𝑠_ 0) 9: **for** _𝑖_ = 2 to _𝑀_ **do** 10: _𝑠_ `candidate` ←A `merger` ( _𝑠_ 0 _, 𝑠_ `init`<sup>_𝜋_(</sup><sup>_𝑖_))</sup> 11: Evaluate _ℎ_ ( _𝑠_ `candidate` ) using D 12: **if** _ℎ_ ( _𝑠_ `candidate` ) ≥ _ℎ_ `best` **then** 13: _𝑠_ 0 ← _𝑠_ `candidate` 14: _ℎ_ `best` ← _ℎ_ ( _𝑠_ 0) 15: **else** 16: **break** 17: **end if** 18: **end for** 

19: **Output** : initial solution _𝑠_ 0 

31 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

#### **B.2. Algorithm for refining a code block for solution improvement** 

**Algorithm 2** Refining solution 

1: **Input** : initial solution _𝑠_ 0, outer loop steps _𝑇_ , inner loop steps _𝐾_ 

2: _𝑠_ `final` ← _𝑠_ 0 

3: _ℎ_ `best` ← _ℎ_ ( _𝑠_ 0) 

4: T `abl` _,_ C = {} _,_ {} 

5: **for** _𝑡_ = 0 to _𝑇_ − 1 **do** 6: _𝑎𝑡_ = A `abl` ( _𝑠𝑡,_ T `abl` ) 7: _𝑟𝑡_ = `exec` ( _𝑎𝑡_ ) 8: T<sup>_𝑡_</sup> `abl`<sup>= A</sup><sup>`summarize`(</sup><sup>_𝑎𝑡, 𝑟𝑡_)</sup> 9: _𝑐𝑡, 𝑝_ 0 = A `extractor` (T `abl`<sup>_𝑡, 𝑠𝑡,_C)</sup> 10: _𝑐𝑡_<sup>0= A</sup><sup>`coder`(</sup><sup>_𝑐𝑡, 𝑝_0)</sup> 11: _𝑠𝑡_<sup>0=</sup><sup>_𝑠𝑡._</sup><sup>`replace`(</sup><sup>_𝑐𝑡, 𝑐_</sup> _𝑡_<sup>0)</sup> 12: Evaluate _ℎ_ ( _𝑠𝑡_<sup>0)using D</sup> 13: **if** _ℎ_ ( _𝑠𝑡_<sup>0)≥</sup><sup>_ℎ_</sup><sup>`best`</sup><sup>**then**</sup> 14: _𝑠_ `final` ← _𝑠𝑡_<sup>0</sup> 15: _ℎ_ `best` ← _ℎ_ ( _𝑠𝑡_<sup>0)</sup> 16: **end if** 17: **for** _𝑘_ = 1 to _𝐾_ − 1 **do** 18: _𝑝𝑘_ = A `planner` ( _𝑐𝑡,_ { _𝑝 𝑗, ℎ_ ( _𝑠𝑡_<sup>_𝑗_)}</sup><sup>_𝑘_</sup> _𝑗_ =<sup>−</sup> 0<sup>1)</sup> 19: _𝑐𝑡_<sup>_𝑘_= A</sup><sup>`coder`(</sup><sup>_𝑐𝑡, 𝑝𝑘_)</sup> 20: _𝑠𝑡_<sup>_𝑘_=</sup><sup>_𝑠𝑡._</sup><sup>`replace`(</sup><sup>_𝑐𝑡, 𝑐_</sup> _𝑡_<sup>_𝑘_)</sup> 21: Evaluate _ℎ_ ( _𝑠𝑡_<sup>_𝑘_)using D</sup> 22: **if** _ℎ_ ( _𝑠𝑡_<sup>_𝑘_)≥</sup><sup>_ℎ_</sup><sup>`best`</sup><sup>**then**</sup> 23: _𝑠_ `final` ← _𝑠𝑡_<sup>_𝑘_</sup> 24: _ℎ_ `best` ← _ℎ_ ( _𝑠𝑡_<sup>_𝑘_)</sup> 25: **end if** 26: **end for** 27: T `abl` ←T `abl` + T `abl`<sup>_𝑡_</sup> 28: C ←C + _𝑐𝑡_ 29: **end for** 

30: **Output** : final solution _𝑠_ `final` 

32 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

#### **B.3. Algorithm for further improvement by exploring ensemble strategies** 

**Algorithm 3** Ensembling final solutions 

1: **Input** : candidate final solutions _𝑠_ `final`<sup>1</sup><sup>_,_· · ·</sup><sup>_, 𝑠_</sup> `final`<sup>_𝐿_, ensemble loop steps</sup><sup>_𝑅_</sup> 2: _𝑒_ 0 = A `ens` _ `planner` ({ _𝑠_ `final`<sup>_𝑙_}</sup> _𝑙_<sup>_𝐿_</sup> =1<sup>)</sup> 3: _𝑠_ `ens`<sup>0= A</sup><sup>`ensembler`(</sup><sup>_𝑒_0</sup><sup>_,_{</sup><sup>_𝑠_</sup> `final`<sup>_𝑙_}</sup> _𝑙_<sup>_𝐿_</sup> =1<sup>)</sup> 4: Evaluate _ℎ_ ( _𝑠_ `ens`<sup>0)using D</sup> 5: **for** _𝑟_ = 1 to _𝑅_ − 1 **do** 6: _𝑒𝑟_ = A `ens` _ `planner` ({ _𝑠_ `final`<sup>_𝑙_}</sup> _𝑙_<sup>_𝐿_</sup> =1<sup>_,_{(</sup><sup>_𝑒𝑗, ℎ_(</sup><sup>_𝑠_</sup> `ens`<sup>_𝑗_)}</sup><sup>_𝑟_</sup> _𝑗_ =<sup>−</sup> 0<sup>1)</sup> 7: _𝑠_ `ens`<sup>_𝑟_= A</sup><sup>`ensembler`(</sup><sup>_𝑒𝑟,_{</sup><sup>_𝑠_</sup> `final`<sup>_𝑙_}</sup> _𝑙_<sup>_𝐿_</sup> =1<sup>)</sup> 8: Evaluate _ℎ_ ( _𝑠_ `ens`<sup>_𝑟_)using D</sup> 9: **end for** 10: _𝑠_ `ens`<sup>∗=</sup><sup>_𝑠_</sup> `ens`<sup>_𝑟_∗where</sup><sup>_𝑟_∗= arg max</sup><sup>_𝑟_∈{0</sup><sup>_,...,𝑅_−1}</sup><sup>_ℎ_(</sup><sup>_𝑠_</sup> `ens`<sup>_𝑟_)</sup> 11: **Output** : _𝑠_ `ens`<sup>∗</sup> 

33 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

### **C. Qualitative examples** 

#### **C.1. Generated code for ablation study** 

We provide an example generated code for ablation study (which is generated by A `abl` ) in the supplementary material (see `example_outputs/ablation.py` ). 

#### **C.2. Raw output of ablation study after execution** 



<!-- Start of picture text -->
[LightGBM] [Info] Number of positive: 2854, number of negative: 2709<br>[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.001167 seconds.<br>You can set `force_col_wise=true` to remove the overhead.<br>[LightGBM] [Info] Total Bins 1647<br>[LightGBM] [Info] Number of data points in the train set: 5563, number of used features: 26<br>[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.513033 -> initscore=0.052142<br>[LightGBM] [Info] Start training from score 0.052142<br>Baseline Validation Performance: 0.8195542774982028<br>[LightGBM] [Info] Number of positive: 2854, number of negative: 2709<br>[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.002991 seconds.<br>You can set `force_col_wise=true` to remove the overhead.<br>[LightGBM] [Info] Total Bins 1647<br>[LightGBM] [Info] Number of data points in the train set: 5563, number of used features: 26<br>[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.513033 -> initscore=0.052142<br>[LightGBM] [Info] Start training from score 0.052142<br>Ablation 1 (No StandardScaler) Validation Performance: 0.8102084831056794<br>[LightGBM] [Info] Number of positive: 2854, number of negative: 2709<br>[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000367 seconds.<br>You can set `force_col_wise=true` to remove the overhead.<br>[LightGBM] [Info] Total Bins 1609<br>[LightGBM] [Info] Number of data points in the train set: 5563, number of used features: 7<br>[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.513033 -> initscore=0.052142<br>[LightGBM] [Info] Start training from score 0.052142<br>Ablation 2 (No OneHotEncoder) Validation Performance: 0.7886412652767792<br>[LightGBM] [Info] Number of positive: 2854, number of negative: 2709<br>[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.001942 seconds.<br>You can set `force_col_wise=true` to remove the overhead.<br>[LightGBM] [Info] Total Bins 1647<br>[LightGBM] [Info] Number of data points in the train set: 5563, number of used features: 26<br>[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.513033 -> initscore=0.052142<br>[LightGBM] [Info] Start training from score 0.052142<br>Ablation 3 (No Imputation) Validation Performance: 0.8195542774982028<br>Final Validation Performance: 0.8195542774982028<br><!-- End of picture text -->

Figure 23 | Example output after running the code for ablation study. 

We provide an example output after running a code for ablation study using spaceship-titanic competition. As shown in Figure 23, the execution result often contains content unrelated to the ablation ( _e.g._ , training information of LightGBM). Therefore, MLE-STAR utilizes A `summarize` to parse the ablation study results, which will be illustrated in the following Appendix C.3. 

34 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

#### **C.3. Summary of ablation study** 

**The ablation study investigated the impact of three preprocessing steps on the performance of a LightGBM classifier: StandardScaler, OneHotEncoder, and Imputation. The baseline model, which included all three preprocessing steps, achieved a validation accuracy of 0.8196.** 

***   **Ablation 1 (No StandardScaler):** Removing the StandardScaler resulted in a slightly lower validation accuracy of 0.8102. This suggests that scaling the numerical features does contribute positively to the model's performance, although the impact is relatively small.** 

***   **Ablation 2 (No OneHotEncoder):** Removing the OneHotEncoder led to a more significant drop in validation accuracy to 0.7886. This indicates that encoding the categorical features using OneHotEncoder is important for the model's ability to learn from the data.** 

***   **Ablation 3 (No Imputation):** Removing the Imputation did not change the validation accuracy, which remained at 0.8196. This suggests that the imputation strategy used (mean for numerical and mode for categorical) does not hurt the model performance.** 

**In summary, the OneHotEncoder has the most significant positive impact on the model's performance, followed by the StandardScaler. The Imputation does not seem to have a significant impact on the model's performance.** 

#### Figure 24 | Example of summarized result of ablation study. 

To parse the information only about the impact of each ML components, MLE-STAR leverages A `summarize` to summarize the raw output of ablation study such as Figure 23. As a result, we obtain the well-organized summary of the ablation study as shown in Figure 24. Note that such summarization is used as input of A `extractor` to extract the code block which has most significant impact on performance. 

### **D. Qualitative comparison** 

We provide qualitative comparison results ( _i.e._ , the final solution code) in `https://github.com/ jaehyun513/MLE-STAR` (see `example_final_solutions/` directory). Solutions generated by MLE-STAR is denoted as `mle_star.py` and solutions generated by AIDE (Jiang et al., 2025) is denoted as `aide.py` in folder name with competition ID. Note that both agent used Gemini-2.0-Flash as a base LLM. 

35 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

### **E. Benchmark** 

#### **E.1. MLE-bench Lite** 

Table 7 <u>|</u> Competitions contained in MLE-bench Lite (Chan et al., 2025). 

|**Competition ID**|**Category**|**Dataset Size (GB)**|
|---|---|---|
|aerial-cactus-identifcation|Image Classifcation|0.0254|
|aptos2019-blindness-detection|Image Classifcation|10.22|
|denoising-dirty-documents|Image To Image|0.06|
|detecting-insults-in-social-commentary|Text Classifcation|0.002|
|dog-breed-identifcation|Image Classifcation|0.75|
|dogs-vs-cats-redux-kernels-edition|Image Classifcation|0.85|
|histopathologic-cancer-detection|Image Regression|7.76|
|jigsaw-toxic-comment-classifcation-challenge|Text Classifcation|0.06|
|leaf-classifcation|Image Classifcation|0.036|
|mlsp-2013-birds|Audio Classifcation|0.5851|
|new-york-city-taxi-fare-prediction|Tabular|5.7|
|nomad2018-predict-transparent-conductors|Tabular|0.00624|
|plant-pathology-2020-fgvc7|Image Classifcation|0.8|
|random-acts-of-pizza|Text Classifcation|0.003|
|ranzcr-clip-catheter-line-classifcation|Image Classifcation|13.13|
|siim-isic-melanoma-classifcation|Image Classifcation|116.16|
|spooky-author-identifcation|Text Classifcation|0.0019|
|tabular-playground-series-dec-2021|Tabular|0.7|
|tabular-playground-series-may-2022|Tabular|0.57|
|text-normalization-challenge-english-language|Seq->Seq|0.01|
|text-normalization-challenge-russian-language|Seq->Seq|0.01|
|the-icml-2013-whale-challenge-right-whale-redux|Audio Classifcation|0.29314|



In this paper, we utilize MLE-bench (especially Lite version) (Chan et al., 2025) as our main benchmark to verify MLE-STAR’s effectiveness compared to the alternatives. In a nutshell, MLEbench consists of 75 offline Kaggle competitions. Each competition has an associated description, dataset, and grading code. Additionally, MLE-bench consists of various problem types, such as tabular prediction, text classification, image classification, etc. However, since utilizing full 75 competitions is expensive, we use the Lite version, which is the low complexity split of MLE-bench ( _i.e._ , MLE-bench Lite). MLE-bench Lite consists of 22 competitions, and the description of competitions is provided in Table 7. 

#### **E.2. Tabular tasks from DS-Agent** 

Table 8 | Tabular competitions used in DS-Agent (Guo et al., 2024) 

|**Competition ID**|**Category**|**Evaluation Metrics**|
|---|---|---|
|media-campaign-cost|Tabular Regression|RMLSE|
|wild-blueberry-yield|Tabular Regression|MAE|
|spaceship-titanic|Tabular Classifcation|Accuracy|
|enzyme-substrate|Tabular Classifcation|AUROC|



We also provide the descriptions of tabular competitions used in DS-Agent’s development phase (Guo et al., 2024) in Table 8. 

36 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

#### **E.3. Generating submission file** 

- **# Introduction - You are a Kaggle grandmaster attending a competition. - In order to win this competition, you need to come up with an excellent solution in Python. - We will now provide a task description and a Python solution. - What you have to do on the solution is just loading test samples and create a submission file. # Task description {task description} # Python solution {final solution} # Your task - Load the test samples and create a submission file. - All the provided data is already prepared and available in the `./input` directory. There is no need to unzip any files. - Test data is available in the `./input` directory. - Save the test predictions in a `submission.csv` file. Put the `submission.csv` into `./final` directory. - You should not drop any test samples. Predict the target value for all test samples. - This is a very easy task because the only thing to do is to load test samples and then replace the validation samples with the test samples. Then you can even use the full training set! # Required - Do not modify the given Python solution code too much. Try to integrate test submission with minimal changes. - There should be no additional headings or text in your response. - The code should be a single-file Python program that is self-contained and can be executed as-is. - Your response should only contain a single code block. - Do not forget the ./final/submission.csv file. - Do not use exit() function in the Python code. - Do not use try: and except: or if else to ignore unintended behavior.** 

Figure 25 | Prompt used for incorporating loading test sample and generating a submission file. 

In order to evaluate on MLE-bench Lite, one should create a submission file about prediction results on test samples with required format. To achieve this, MLE-STAR uses an agent A `test` , which takes the task description and the final solution as input, and outputs the code that incorporates loading test sample and creating a submission file. This is done by using a prompt in Figure 25. 

37 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

##### **# Introduction** 

- **From the give Python solution, you need to extract a code block where subsampling of training samples is used. We will now provide the current Python solution."** 

- **# Current Python solution {final solution} # Your task - Extract a code block where subsampling of training samples is used.** 

- **# Response format - Your response should be a single markdown code block (wrapped in ```) which is the code block.** 

- **The code block should be exactly extracted from the Python script provided above.** 

Figure 26 | Prompt used for extracting the code block which performs subsampling. 

##### **# Introduction** 

- **From the give Python code block, remove the subsampling and make it to use full training samples. We will now provide the current Python code block.** 

- **# Current Python code block {code block with subsampling} # Your task** 

- **Remove the subsampling and make it to use full training samples. - Note that all the variable including actual data is defined earlier (since you are just seeing a code block), therefore do not introduce dummy variables. # Response format** 

- **Your response should be a single markdown code block (wrapped in ```) which is the code block.** 

Figure 27 | Prompt used for guiding MLE-STAR to utilizie full training samples. 

**Removing subsampling.** As shown in Figure 10, MLE-STAR uses the subset of training sample for faster refinement (since evaluating the solution candidate can take a lot of time). However, in order to get a better performance, when generating a submission file MLE-STAR removes such subsampling code. Specifically, this is done by first extracting the code block which performs subsampling (using prompt in Figure 26), and then modify the extracted code block to utilize all the provided samples, using prompt in Figure 27. 

### **F. Experimental setup** 

We conducted our experiments mainly using 96 vCPUs with 360 GB Memory (Intel(R) Xeon(R) CPU), and 8 NVIDIA V100 GPUs with 16 GB Memory. 

**Required time to generate a single solution using MLE-STAR.** With the configuration of four retrieved models, four inner loops, four outer loops, and five rounds for exploring the ensemble strategy, MLE-STAR requires 14.1 hours to generate a single final solution, on average across 22 tasks and all three random trials ( _i.e._ , total 66 experiments). On the other hand, we found that AIDE (Jiang et al., 2025) requires 15.4 hours. This indicates that our method does not require more time to run compare to the best alternative. Note that a maximum time limit of 24 hours was set for both methods, following the MLE-bench’s experimental setup. 

38 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

### **G. Additional quantitative results** 

Table 9 | Additional comparisons with AutoGluon and DS-Agent in four tabular tasks. 

|Model|media-campaign-cost|wild-blueberry-yield|spaceship-titanic|enzyme-substrate|
|---|---|---|---|---|
|Evaluation Metrics|RMLSE (↓)|MAE (↓)|Accuracy (↑)|AUROC (↑)|
|**AutoGluon (Erickson et al., 2020)**|0.2707|305|0.8044|0.8683|
|**DS-Agent (Guo et al., 2024)**|||||
|gpt-3.5|**0.2702**|291|/|0.5534|
|gpt-4|0.2947|267|0.7977|0.8322|
|gemini-2.0-fash|0.2964|213|0.7982|0.8727|
|**MLE-STAR (Ours)**|||||
|**gemini-2.0-fash**|0.2911|**163**|**0.8091**|**0.9101**|



This section provides detailed results for the comparison with DS-Agent (Guo et al., 2024). In particular, we provide additional comparisons with AutoGluon (Erickson et al., 2020) and DS-Agent using other LLMs ( _i.e._ , GPT-3.5 and GPT-4). Except for DS-Agent with Gemini-2.0-Flash and MLESTAR, all experimental results are taken from the original paper (Guo et al., 2024). As shown in Table 9, MLE-STAR consistently outperforms DS-Agent with Gemini-2.0-Flash, while also outperforms AutoGluon with high margin on three tabular tasks. 

It is worth to note that AutoGluon is restricted to task types, _i.e._ , specially designed for tabular data. In contrast, MLE-STAR is a general framework for any kinds of tasks, where well-written task description, containing the task information, is the only requirement to work on the given tasks. Therefore, while AutoGluon is not a direct competitor in this regard, MLE-STAR shows improved performance even when compared to AutoGluon. 

39 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

### **H. Analysis on data contamination** 

**Your task is to check whether the python solution is similar to the reference discussion. Now we will give you reference discussion and our python solution.** 

##### **# Reference discussion** 

**{reference discussion}** 

##### **# Python solution** 

**{final solution}** 

##### **# Your task** 

**- Check whether the python solution just copy and pastes the reference discussion.** 

- **If it is sufficiently novel and different, please answer 'Novel'.** 

- **Otherwise, if you think it is too similar, please answer 'Same'.** 

- **Your answer should be only one of 'Novel' or 'Same'.** 

Figure 28 | Prompt used for identifying whether the final solution generated by MLE-STAR is novel. 

Since Kaggle competitions in MLE-bench are publicly accessible, there is a potential risk that LLMs might have been trained with the relevant discussions about the challenge. For example, if an LLM has memorized a discussion of the best performing solution, one easy way for the MLE agent to follow that discussion during the refinement phase. 

However, to alleviate such potential issue, we show that MLE-STAR’s solution is sufficiently novel compared to the discussions on Kaggle. Here, we use discussions collected in GibHub repository of MLE-bench (Chan et al., 2025) are collected by the authors of MLE-bench (Chan et al., 2025). To be specific, these discussions are top discussion posts of each competition. As a result, we collected a total of 25 discussions from 7 competitions, resulting in 75 discussion-solution pairs, where solution represents the final solution obtained by MLE-STAR. Using LLM as a judge with the prompt in Figure 28, we found that all the final solutions generated by MLE-STAR with Gemini-2.0-Flash were judged to be sufficiently novel compared to the top discussions. Note that we use Gemini-2.5-Pro to judge the novelty of MLE-STAR’s solutions. 

### **I. Broader impacts** 

By automating complex ML tasks, MLE-STAR could lower the barrier to entry for individuals and organizations looking to leverage ML, potentially fostering innovation across various sectors. In addition, as state-of-the-art models are updated and improved over time, the performance of solutions generated by MLE-STAR is expected to be automatically boosted. This is because our framework leverages a search engine to retrieve effective models from the web to form its solutions. This inherent adaptability ensures that MLE-STAR continues to provide increasingly better solutions as the field of ML advances. 

40 

MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 

### **J. Related works on data science agents** 

While our work focuses on LLM-based agents tailored for machine learning engineering, other research explores agents for general data science tasks (Hu et al., 2024; Huang et al., 2024b; Jing et al., 2025), including data analysis and visualization. Among these, Data Interpreter (Hong et al., 2024) employs a graph-based approach, dividing tasks into subtasks and refining the task graph based on successful completion. DatawiseAgent (You et al., 2025) proposes a two-stage process: initially generating a tree-structured plan, followed by an exploration of the solution space. Although these methods exhibit generalizability to various data science tasks, including aspects of machine learning engineering, their evaluation prioritizes overall task completion rates rather than performance on specific engineering challenges. 

41 

