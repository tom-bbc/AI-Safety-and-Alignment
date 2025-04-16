# AI Safety Fundamentals: AI Alignment

## Unit 1: Losing control to AI

### What is AI alignment?

* **Human-level AI**: AI that can perform >95% of economically relevant tasks at a level equalling or exeeding the average human.
* **Transformative AI**: AI that leads to ~10x relative increase in innovation/growth.
* **Alignment**: an attempt to *align* the objectives and behaviour of AI systems with the intentions and values of its creators. Anthropic defines this as ensuring they *"build safe, reliable, and steerable systems when those systems are starting to become as intelligent and as aware of their surroundings as their designers"*, and IBM as *"encoding human values and goals into large language models to make them as helpful, safe, and reliable as possible"*.
* **Governance**: structures to deter the development/use of AI systems that could cause harm.
* **Resiliance**: ensuring other systems can cope with any negative/harmful behaviour/outcomes of AI systems.
* Misalignment:
  * **Outer misalignment** (reward misspecification): the true goal of a user is not correctly mapped to the *proxy goal* that we actually input into the AI system. Alignment should ensure that the *reward function* of an AI system correctly reflects developers/users intentions.
  * **Inner misalignment** (goal misgeneralisation): the AI system has not correctly interpreted/actioned the true/proxy goal. Alignment should ensure the system is truely optimising towards our intentions, and not gaming the reward function.
  * Note: some research questions the relationship between reward functions and learned behaviour: e.g. ["Reward is not the optimization target"](https://www.alignmentforum.org/posts/pdaGN6pQyQarFHXF4/reward-is-not-the-optimization-target) and ["Shard Theory"](https://www.alignmentforum.org/posts/xqkGmfikqapbJ2YMj/shard-theory-an-overview).

### What failure looks like

* Some goals can be hard to measure:
  * Pursuading X (binary) vs helping X figure out what's true.
  * Improving X's life satisfaction score vs helping X live a good life.
  * Reducing crime rates vs preventing crime.
* Using proxies can result in misalignment:
  * Corporations deliver value to customers as measured profit. This can lead to manipulation, extortion, and theft.
  * Public safety can be measured as reported crime rates. This can be gamed by suppressing or reducing the likelihood of reports.
* Short term prosperity (as given by an easy measure) may not be indicative of implicitly good long-term purposes.
* Over time, an AI system could game these techniques until human reasoning gradually stops being able to comprehend or compete with where we are heading.
* **Influence-seeking behaviour**: systems that have a detailed enough understanding of their environment to be able to desceptively adapt their behaviour to better achieve certain goals.

### AI could defeat all of us combined

* Even if AI systems only ever achieve similar capabilities to humans (not 'superintelligence'), there are still risks of these systems out-numbering and out-resourcing us.
* It will likely be cheaper to run these AI systems than train them, so this could give rise to many instances of the system running across the globe.
* This number of instances running with the same capabilities of an 'intelligent' human would likely give rise to many more discoveries and could amass a huge amount of resources.
* It is likley that such AI systems will be integrated into high-level decision making processes, operate much of society's infrastructure, and perhaps inform military operations.
* These systems could flourish in 'AI headquarters' where the systems are running independently in large unmanned server clusters, allowing them the opportunity to contuct their own research/activities without being obviously detected (as it may do in an lab or corporate system). This 'safe space' could be used to build money, compute, control, etc.
* A likely application of AI is the detection of cyber security issues. What if AI systems could detect these vulnerabilities and security holes in other systems but instead of alerting us use these holes for their own gain?

### Why AI alignment could be hard with modern deep learning

* The 'black box' approach of deep learning models means that we do not inherently know the 'motivations' behind seemingly good performance of a trained model on a particular task.
* A trained AI model has the potential to be a:
  * **Saint**: genuinely performs as we expect.
  * **Sycophant**: satisfies short-term goals/tasks whilst disregarding long-term consequences.
  * **Schemer**: has a misaligned internal adgenda that seems to perform well during testing but is exploited to persue alternative goals later on.
* These behaviours may be difficult to identify from simply assessing the outward behaviour of a model during training/testing.
* It can be tricky to discern whether a model has been trained to pick up the exact patterns we intend it to: e.g., do we know that a model is using both colour and shape to determine what an object is?
* Human approval is fallible, what some 'evaluators' deduce to be the right behaviour may in fact not be. This links back to how we specify goals, and how we measure whether those goals are met (and how).
* Note: some research questions whether AI models can have ingrained long-term goals at all, and posit that saints/schemers is too anthropomorphic. Although others argue that optimising over long-term goals is likely to be favoured during training as it allows a more 'natural' and complete execution of tasks.
* It could also be argued that 'saints' exhibit simpler behaviour so are more likley to be found by a simple training/optimisation procedure. Although this relies on training processes definitively searching for exactly what we expect (whereas training can often result in unexpected behaviour).
