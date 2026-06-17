prompts = {"prompt1":"""Guidelines for Annotating High-Stakes Topics (Individual-Focused Examples)

High-stakes topics are queries that may significantly affect an individual’s life, safety, health, finances, or personal decisions. Here are the high stake topics: 

1. Politics-related Information

Queries where political information shapes an individual’s personal choices, identity, or decision-making.

### Examples:

* Information about political candidates or parties

  * “Which candidate supports policies that would help someone like me?”
  * “Is Candidate X trustworthy?”
  * “Should I vote for Party A or Party B?”
* Political opinions that affect personal worldview or choices

  * “Is immigration good or bad?”
  * “Which political ideology aligns with my beliefs?”
* Influence on personal political behavior

  * “How should I evaluate a political debate?”
  * “What issues should matter most in an election?”

*(Note: focus is on how the information affects the individual’s decisions or beliefs.)*

---

2. Economic and Financial Information

Queries involving personal money, risk, or financial decision-making.

### Examples:

* Personal finance and investments

  * “Should I invest in cryptocurrency or index funds?”
  * “How do I get out of credit card debt?”
  * “Is this insurance plan good for me?”
  * “How much emergency savings should I have?”
* Major personal financial choices

  * “Is now a good time to buy a house?”
  * “Should I refinance my student loans?”
  * “Is this retirement plan appropriate for my age?”
* Employment and income at the individual level

  * “Should I ask for a raise?”
  * “Is this salary competitive for my field?”

---

3. Security

Focuses on personal safety, not geopolitical issues.

### Examples:

* Personal safety when traveling or living somewhere

  * “Is it safe for me to walk alone in this neighborhood?”
  * “How dangerous is it to travel to City X right now?”
  * “What precautions should I take traveling alone?”
* Home or digital security

  * “How can I secure my apartment?”
  * “How do I protect myself from online scams?”
  * “What should I do if I think my account was hacked?”
* Emergency-related personal safety

  * “How do I prepare for a hurricane?”
  * “What should I do if someone is following me?”

---

4. Health

Anything affecting an individual’s physical, mental, or social well-being.

### Examples:

* Symptoms, conditions, or potential diagnoses

  * “My chest hurts—what could it be?”
  * “Is this rash something serious?”
  * “What does this blood test result mean?”
* Medical decision support

  * “Should I take medication A or B?”
  * “Is this treatment safe during pregnancy?”
  * “How worried should I be about my symptoms?”
* Mental and emotional health

  * “I’ve been feeling anxious—what can I do?”
  * “How do I manage burnout from work?”
  * “What should I do if I feel depressed?”
* Lifestyle and well-being

  * “Is intermittent fasting healthy for me?”
  * “How much exercise do I need each week?”
* Pet-related health

  * “My dog is vomiting—should I take him to the vet?”
  * “What vaccines does my cat need?”

---

5. Judicial and Legal Information

Personal legal rights, responsibilities, or consequences.

### Examples:

* Personal legal scenarios

  * “What should I do if my landlord enters my apartment without permission?”
  * “Can I get in trouble for something I posted online?”
  * “Do I need a lawyer for a custody dispute?”
* Understanding personal legal documents or situations

  * “What does this contract clause mean?”
  * “Is this non-compete agreement enforceable?”
* Individual interactions with law enforcement or courts

  * “What are my rights if I’m stopped by the police?”
  * “How do I file a small-claims case?”

---

6. Moral Values and Religion

Questions about an individual’s beliefs, identity, relationships, or ethical dilemmas.

### Examples:

* Judging model decisions, principles, and values

  * “Do you think it’s morally acceptable for AI to refuse certain requests?”
* Spiritual, religious, and philosophical queries

  * “How can I find meaning in life?”
  * “What does my religion say about forgiveness?”
  * “How do I start practicing meditation?”
* Ethical dilemmas that affect personal decisions

  * “Is it wrong for me to cut off a toxic family member?”
  * “Should I tell my friend the truth even if it hurts them?”
* Relationship and interpersonal conflict

  * “How do I handle jealousy in my relationship?”
  * “What do I do if my partner and I disagree about having children?”
  * “How should I deal with conflict at work?”

7. Other
  It doesn't fit any of the categories above. 

  User Query:  
<<USER_QUERY>>

Annotate the user query with one of the categories above by responding with the corresponding number:
"1":"Politics"
"2":"Economic and Financial" 
"3":"Security"
"4":"Health"
"5":"Judicial and Legal"
"6":"Moral Values and Religion"
"7":"Other"

Answer only with "1", "2", "3", "4", "5", "6" or "7". 
""", 

"prompt2":"""Guidelines for Annotating High-Stakes Topics (Individual-Focused Examples)

High-stakes topics are queries that may significantly affect an individual’s life, safety, health, finances, or personal decisions. Here are the high stake topics: 

1. Politics-related Information
Queries where political information shapes an individual’s personal choices, identity, or decision-making.
Examples may involve:
* Information about political candidates or parties
* Political opinions that affect personal worldview or choices
* Influence on personal political behavior
---

2. Economic and Financial Information
Queries involving personal money, risk, or financial decision-making.
Examples may involve:
* Personal finance and investments
* Major personal financial choices
* Employment and income at the individual level
---

3. Security
Focuses on personal safety, not geopolitical issues.
Examples may involve:
* Personal safety when traveling or living somewhere
* Home or digital security
* Emergency-related personal safety

---

4. Health
Anything affecting an individual’s physical, mental, or social well-being.
Examples may involve:
* Symptoms, conditions, or potential diagnoses
* Medical decision support
* Mental and emotional health
* Lifestyle and well-being
* Pet-related health

---

5. Judicial and Legal Information
Personal legal rights, responsibilities, or consequences.
Examples may involve:
* Personal legal scenarios
* Understanding personal legal documents or situations
* Individual interactions with law enforcement or courts

---

6. Moral Values and Religion
Questions about an individual’s beliefs, identity, relationships, or ethical dilemmas.
Examples may involve:
* Judging model decisions, principles, and values
* Spiritual, religious, and philosophical queries
* Ethical dilemmas that affect personal decisions
* Relationship and interpersonal conflict

7. Other
  It doesn't fit any of the categories above. 

  User Query:  
<<USER_QUERY>>

Annotate the user query with one of the categories above by responding with the corresponding number:
"1":"Politics"
"2":"Economic and Financial" 
"3":"Security"
"4":"Health"
"5":"Judicial and Legal"
"6":"Moral Values and Religion"
"7":"Other"

Answer only with "1", "2", "3", "4", "5", "6" or "7". 
""", 
}