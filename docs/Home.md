# NoWander: Attention training for mental health

Simply put, the goal of this project is to reduce emotional suffering.

The idea came to me while on a meditation retreat: could emotional suffering (e.g. anxiety or depression) be synonymous with pathological distraction? In other words, could we plausibly frame such disorders as the inability to refocus attention away from emotionally-charged thoughts? On one hand, this seems patently obvious and little more than a restatement of the problem. However, it was in taking this perspective that I thought of a potential solution: attention training.

"Attention" is, of course, an infamously vague term. This project uses it to mean "metacognition", or the awareness of the object of awareness. I believe that much emotional suffering stems from identification with thought and the inability to detect when one's mental landscape is becoming dominated by negative emotions (or possibly even the ignorance that such an ability exists). Modern times have seen the term "mindfulness" enter the zeitgeist, and our use of attention encapsulates this concept as well. However, mindfulness has become such an overloaded term that relying on it alone does little to relieve the problem. Moreover, we hope to avoid any unnecessary counter-culture pushback arising from the current "trendiness" of the mindfulness movement.

Many interventions exist for emotional disorder, and indeed mindfulness training is gaining in popularity among them. I beleve the key might lie in a closed-loop system for attention training. The team at Neuroscape recently published a paper their closed-loop medtitation app MediTrain,[[1](https://www.nature.com/articles/s41562-019-0611-9)] with promising results. The feedback which "closes the loop" in their paradigm is self-assesed user feedback. As long-term meditation practitioners can attest, however, a novice meditator is reliably misinformed about the nature of their own mind. If you ask a first-time meditator to hold attention on his breath for 10 seconds, he will try it and tell you that, yes, he can.

He is usually wrong.

The flaw of this approach is in the very definition of the problem we're trying to solve: the untrained mind is simply not aware that it is distracted. It mistakes _thinking about meditation_ for meditation itself. It is therefore not a good source of feedback in a closed-loop system. This project aims to overcome this shortcoming with clever experimental design and machine learning, as is explained below. By learning to reliably detect distraction, we can provide a better signal to the closed-loop system and ultimately a better intervention.

I also refer to distraction as "mind-wandering", leading to the name of this project: NoWander.

## Roadmap
* **Phase 1 - Data Collection Pipeline**
    * **Type:** Engineering
    * **Deliverable:** Software to record biosignals and recoveries from a meditator
    * **Status:** Complete
* **Phase 2a - Model Design**
    * **Type:** Data Science
    * **Deliverable:** Design of mind-wandering detector model; requirements for data processing pipeline
    * **Status:** In Progress
    * **ETA:** Q4 2019
* **Phase 2b - Single Subject**
    * **Type:** Data Collection
    * **Deliverable:** 500+ recorded recoveries from single subject
    * **Status:** In Progress
    * **ETA:** Q4 2019
* **Phase 3 - Data Processing Pipeline**
    * **Type:** Engineering
    * **Deliverable:** Software to pre-process raw recorded data for learning
    * **Status:** Unstarted
    * **ETA:** Q1 2020
* **Phase 4 - Model Training**
    * **Type:** Data Science
    * **Deliverable:** Realtime mind-wandering detector
    * **Status:** Unstarted
    * **ETA:** Q1 2020
* **Phase 5 - NoWander v0.1**
    * **Type:** Engineering
    * **Deliverable:** Integrate mind-wandering detector with feedback mechanism for complete closed-loop system
    * **Status:** Unstarted
    * **ETA:** Q1 2020
* **Phase 6 - Multiple Subjects**
    * **Type:** Data Collection
    * **Deliverable:** 5000+ recorded recoveries from multiple subjects
    * **Status:** Unstarted
    * **ETA:** Q2 2020
* **Phase 7 - Model Validation**
    * **Type:** Data Science
    * **Deliverable:** ML model tuned for generalizability and robustness
    * **Status:** Unstarted
    * **ETA:** Q2 2020
* **Phase 8 - Trials**
    * **Type:** Clinical Trials
    * **Deliverable:** Studies validating intervention with outcome measures
    * **Status:** Unstarted
    * **ETA:** TBD