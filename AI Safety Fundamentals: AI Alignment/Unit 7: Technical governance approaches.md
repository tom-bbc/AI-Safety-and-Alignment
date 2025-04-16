# AI Safety Fundamentals: AI Alignment

## Unit 7: Technical governance approaches

### Emerging processes for frontier AI safety

#### Responsible capability scaling

* Risk assessments and continual monitoring.
* 'Risk thresholds' that limit 'acceptable' risk.
* Mitigations adapted to varying stages/contexts of deployment.
* Ability to pause development/deployment.
* Internal accountability and governance.

#### Model evaluations and red teaming

* Model evaluations measure traits of AI systems quantitatively.
* Evaluations should check potential sources of risk incl. model capabilities, controllability, society harms, and system security.
* Red teaming observes an AI system from an adversarial perspective to determine potential areas of compromise/misuse.
* Evaluation and red teaming should be done across multiple stages of the model lifecycle incl. training and deployment.

#### Model reporting and information sharing

* Transparency can help identify AI benefits and risks, conduct AI governance, and increase public trust.
* This includes model-agnostic information, such as risk assessment steps, and model-specific information, including training and deployment considerations.

#### Security controls incl. securing model weights

* Cyber security processes should underpin the safety, reliability, predictability, ethics, and regulatory compliance of AI systems.
* Security measures should be implemented across underlying infrastructure and supply chains.
* Protect assets within the system such as model weights and training data.
* Monitoring of system behaviour is key to identify potential attacks.

#### Reporting structure for vulnerabilities

* 'Vulnerabilities' are unidentified safety and security issues that persist within an AI system.
* Vulnerability management processes should allow outsiders to report any identified vulnerabilities.

#### Identifiers of AI generated material

* Adopting multiple identifier mechanisms could help mitigate some risks associated with the distribution of disceptive AI generated content, bias in AI content, and loss of trust in digital information sources.
* However, authentication solutions currently under development, such as watermarking tools and AI output databases, cannot currently entirely miticate the risks of AI generated content.

#### Preventing and monitoring model misuse

* Monitor for common ways models are misused and safeguards circumvented.
* Model input/output filters.
* Measures to prevent harmful outputs, such as prompting, fine-tuning, and rejection sampling.
* Enabling the potential for rapid model rollback and withdrawl.

#### Data input controls and audits

* Training AI systems on poor quality data can increase model risks by enhancing potentially dangerous capabilities.
* Datasets sould be controlled/audited to minimise these risks.


### Computing Power and the Governance of AI

* **Compute governance**: using the supply, measurement, and monitoring of computing resources/power ('compute') as a mechanism to persue AI governance policies.
* This can be implemented as:
  * Monitoring of compute to gain visibility of AI development and use.
  * Shaping the allocation of resources by subsidising/limiting compute.
  * Compute 'guardrails' to enforce regulations.
* Compute is inherently *detectable* (chip supply/usage can be traced), *excludable* (supply can be limited), and *quantifiable* (chips/power can be measured).
* Model performance on training objectives such as next token prediction seem to obey scaling laws w.r.t. the amount of compute (FLOPs) and data used. Because of this, the compute used to train AI systems has roughly doubled every six months since 2010.
*
