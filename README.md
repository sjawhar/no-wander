# NoWander: Attention training for mental health

Simply put, the goal of this project is to reduce emotional suffering.

The idea came to me while on a meditation retreat: could emotional suffering (e.g. anxiety or depression) be synonymous with pathological distraction? In other words, could we plausibly frame such disorders as the inability to refocus attention away from emotionally-charged thoughts? On one hand, this seems patently obvious and little more than a restatement of the problem. However, it was in taking this perspective that I thought of a potential solution: attention training.

"Attention" is, of course, an infamously vague term. This project uses it to mean "metacognition", or the awareness of the object of awareness. I believe that much emotional suffering stems from identification with thought and the inability to detect when one's mental landscape is becoming dominated by negative emotions (or possibly even the ignorance that such an ability exists). Modern times have seen the term "mindfulness" enter the zeitgeist, and our use of attention encapsulates this concept as well. However, mindfulness has become such an overloaded term that relying on it alone does little to relieve the problem. Moreover, we hope to avoid any unnecessary counter-culture pushback arising from the current "trendiness" of the mindfulness movement.

Many interventions exist for emotional disorder, and indeed mindfulness training is gaining in popularity among them. I beleve the key might lie in a closed-loop system for attention training. The team at Neuroscape recently published a paper their closed-loop medtitation app MediTrain,[[1](https://www.nature.com/articles/s41562-019-0611-9)] with promising results. The feedback which "closes the loop" in their paradigm is self-assesed user feedback. As long-term meditation practitioners can attest, however, a novice meditator is reliably misinformed about the nature of their own mind. If you ask a first-time meditator to hold attention on his breath for 10 seconds, he will try it and tell you that, yes, he can.

**He is usually wrong.**

The flaw of this approach is in the very definition of the problem we're trying to solve: the untrained mind is simply not aware that it is distracted. It mistakes _thinking about meditation_ for meditation itself. It is therefore not a good source of feedback in a closed-loop system. This project aims to overcome this shortcoming with clever experimental design and machine learning, as is explained below. By learning to reliably detect distraction, we can provide a better signal to the closed-loop system and ultimately a better intervention.

I also refer to distraction as "mind-wandering", leading to the name of this project: NoWander.

## Experimental Design
### Step 1: Learning
In the moment, one can be (and usually is) distracted without realizing it. There is an exception to this, though, which we might call the "recovery". When meditating, one proceeds through several cycles of attempting to hold focus (on the breath, for example), eventually failing and becoming distracted, and then (sometime later) realizing one's distraction and returning to the object of focus. This moment of realization is the "recovery". In that moment, a meditator can be sure that she was previously distracted but is now focused again. If the meditator's biosignals are being recorded along with the times of these recoveries, then for each recovery we have two labeled samples: the moments preceding a recovery are "distraction", and those following (up to a point) are "focus". By repeating this process many times and with many subjects, we collect a dataset from which we might learn the neural correlates of distraction.

### Step 2: Training
Once a reliable method for detecting distraction has been established, we can use it in our closed-loop system. Once again, the subject is asked to meditate while their biosignals are recorded. Now, however, we can detect distraction before the subject is aware of it themselves. When we do, we allow the subject a "grace period" to recover on their own. When the grace period elapses, we provide a subtle queue (e.g. a bell chime) to bring the subject's focus back to the present, and so the cycle repeats. By progressively shortening these "grace periods", we train the subject to recognize and detach from their distraction more quickly. If this training transfers to other aspects of life, I believe it can seriously alleviate emotional suffering.

## Project Roadmap
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


## Other Notes
### Why Meditation?
TODO: No confounding signals from movement, vision, audition, etc. Nothing but cognition.

### Possible Extensions
TODO
* Rely only on phone camera and computer vision (posture changes, eyelid fluttering, heart rate)

### Reflections on Awareness
To understand our use of awareness, consider the following example:
* You are walking to the store. It is cool and sunny, and you're feeling pretty great.
* Lost in considering your grocery list, you accidentally bump into another pedestrian on the narrow sidewalk.
* "Moron," she grumbles, but you both continue on.
* "Wow," you think, "what a mean person!" Anger arises. You find yourself imagining the dressing-down you'd give this person if you were to see her again. You are sure this is a bad person with no friends.

This is pathological distraction. It may feel like this is an active process, that these thoughts are a product of your conscious efforts, that they are _your_ thoughts. In reality, though, you have been swept away by mental sensations over which you have no control. You are distracted, to potentially disastrous effect.

Imagine instead that the following thought had arisen after the encounter: "Huh, I wonder what's on _her_ mind?" Now, you find yourself imagining some personal tragedy that might have befallen her this morning, the trials and tribulations that yet await her, and the heroic (yet romantically tragic) role she is destined to play.

Now, imagine this instead: "Hey, that's a cool jacket! Where did she get it? I wonder, is it vintage? All the good stuff is vintage. She must know a lot about fashion. I wonder if she could teach me a few things, my wardrobe is so plain."

Before encountering this person, you obviously had no opinion of them. By the end of each mental tirade, however, you have concluded something about this person's character. These three conclusions are mutually exclusive, yet which one you reach depends only on the appearance of a single thought.

TODO: Application to identity and emotional states. "I am sad."
