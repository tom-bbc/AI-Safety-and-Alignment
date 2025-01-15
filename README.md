# AI Alignment Fast-Track

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
