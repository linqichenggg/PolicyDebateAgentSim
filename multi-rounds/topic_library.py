"""Political topic pools used for stance-based dialogue simulation."""

debate_topics = [
    "Japan should increase the consumption tax to 15% to stabilize long-term social security funding.",
    "Japan should significantly expand immigration to address labor shortages.",
    "Japan should reduce dependence on nuclear power and accelerate renewable transition.",
    "Japan should revise Article 9 to allow broader collective self-defense operations.",
    "Public universities should be tuition-free for all domestic students.",
    "The government should impose stronger regulation on major social media platforms."
]

topic_sentence_support_tax = [
    "A higher consumption tax is painful, but it secures pension and healthcare sustainability.",
    "Without tax reform, demographic pressure will make social security funding unstable.",
    "Raising the tax gradually is a realistic fiscal strategy.",
    "Short-term burden is acceptable if it prevents long-term system collapse.",
    "A predictable tax increase can be paired with targeted support for low-income households.",
    "Fiscal responsibility requires stable revenue, and consumption tax provides that base.",
    "I support the increase because the aging population trend is structurally clear.",
    "We need a durable funding mechanism, not temporary budget patches.",
    "If the tax rise is predictable, households and firms can adapt more smoothly.",
    "Intergenerational fairness requires stable financing for social programs."
]

topic_sentence_oppose_tax = [
    "A higher consumption tax disproportionately hurts low-income households.",
    "Demand will weaken if households reduce spending under heavier tax pressure.",
    "Tax hikes can slow recovery and widen inequality.",
    "The government should cut waste before increasing the tax burden.",
    "Income-side reforms are fairer than broad consumption tax expansion.",
    "People already face rising living costs, so this policy is mistimed.",
    "I oppose the increase because it shifts structural problems to ordinary consumers.",
    "Fiscal reform should prioritize efficiency and redistribution before raising consumption tax.",
    "A broad consumption tax is regressive unless compensation is exceptionally strong.",
    "Economic confidence may weaken when daily essentials become more expensive."
]

topic_sentence_support_immigration = [
    "Expanded immigration is necessary to sustain labor supply and economic activity.",
    "Many sectors already face chronic worker shortages that domestic labor cannot fill fast enough.",
    "A managed immigration framework can balance growth and social stability.",
    "Demographic decline requires structural openness, not only short-term labor programs.",
    "Immigration can support innovation and productivity when integration policies are strong.",
    "I support expansion because labor shortages are a long-term reality.",
    "With clear rules, immigration strengthens both local communities and industry.",
    "A shrinking workforce needs broader talent inflow.",
    "Regional economies can revive when labor and entrepreneurship are replenished.",
    "Well-designed integration policy can turn demographic pressure into growth potential."
]

topic_sentence_oppose_immigration = [
    "Rapid immigration expansion can strain local services and social cohesion.",
    "Integration policy is not ready for a large-scale intake.",
    "Wage stagnation risk should be considered before expanding labor inflow.",
    "Domestic training and productivity reform should come first.",
    "Without strong governance, expansion could create regional inequality and tension.",
    "I oppose rapid expansion because institutional capacity is still limited.",
    "Policy should prioritize orderly adaptation over sudden demographic shifts.",
    "Labor shortages should be solved with automation and workforce reform first.",
    "Policy pacing matters; abrupt expansion can outstrip schools, housing, and local services.",
    "Long-term social trust can be damaged if integration support is underfunded."
]

topic_sentence_support_energy = [
    "Reducing nuclear dependence is necessary for long-term safety and public trust.",
    "Renewables and grid modernization should be accelerated now, not later.",
    "Energy transition reduces systemic risk from large centralized failures.",
    "A phased reduction is feasible with storage, efficiency, and regional balancing.",
    "I support the transition because social acceptance matters for energy legitimacy.",
    "Renewable investment can create domestic industry and resilience.",
    "Lower nuclear reliance aligns with risk management and policy credibility.",
    "Transition planning should start early to avoid future shocks.",
    "Distributed renewable systems can improve resilience during disasters.",
    "Public trust improves when energy strategy reduces high-consequence accident risk."
]

topic_sentence_oppose_energy = [
    "Rapid nuclear phase-down risks supply instability and higher electricity prices.",
    "Base-load reliability remains a major challenge without sufficient storage capacity.",
    "A balanced mix should keep nuclear as a transitional pillar.",
    "Industrial competitiveness may decline if power costs rise too quickly.",
    "I oppose aggressive reduction because energy security must remain central.",
    "Policy should optimize reliability, affordability, and emissions together.",
    "Replacing stable generation too fast creates operational and fiscal risk.",
    "Transition should be gradual and technology-neutral.",
    "Energy policy should avoid abrupt structural moves that raise import dependence.",
    "Grid stability risk is too high if dispatchable supply is reduced too quickly."
]

topic_sentence_support_article9 = [
    "Security conditions have changed, so legal scope for collective defense should be updated.",
    "Alliance credibility requires clearer operational authorization.",
    "A narrower legal framework may limit response capacity in regional crises.",
    "Revision can include democratic oversight and strict constraints.",
    "I support revision because deterrence requires institutional clarity.",
    "Policy realism requires legal alignment with current security responsibilities.",
    "Defensive coordination with partners needs a stable constitutional basis.",
    "Modern threats demand modern legal tools.",
    "Clearer legal authority can improve crisis coordination and deterrence credibility.",
    "Legal modernization can coexist with strict civilian oversight and transparency."
]

topic_sentence_oppose_article9 = [
    "Expanding collective defense authority risks long-term militarization.",
    "Article 9 has been a core normative foundation and should be preserved.",
    "Legal revision may increase regional distrust and escalation risk.",
    "Security can be strengthened through diplomacy and de-escalation frameworks.",
    "I oppose revision because constitutional restraint remains strategically valuable.",
    "Policy change in this area should require overwhelming public consensus.",
    "Risk control should prioritize conflict prevention, not force expansion.",
    "Once expanded, security powers are difficult to roll back.",
    "Constitutional restraint has strategic value in reducing regional arms pressure.",
    "A preventive peace posture can be weakened by broadened military authorization."
]

topic_sentence_support_tuition = [
    "Tuition-free public universities improve equal opportunity and social mobility.",
    "Access to higher education should depend less on family income.",
    "Long-term productivity benefits justify public investment in education.",
    "Financial barriers currently suppress talent development and innovation capacity.",
    "I support universal tuition-free access as a structural competitiveness policy.",
    "The policy can be phased with fiscal safeguards and quality standards.",
    "Education funding is an investment, not only a cost item.",
    "Broader access strengthens future tax base and civic outcomes.",
    "Removing tuition barriers can widen participation from underrepresented groups.",
    "Human capital investment yields compounding long-run public returns."
]

topic_sentence_oppose_tuition = [
    "Universal tuition-free policy may be fiscally unsustainable without major trade-offs.",
    "Targeted aid is more efficient than blanket subsidy.",
    "Policy design should prioritize students with actual financial need.",
    "Without quality safeguards, free tuition alone does not guarantee better outcomes.",
    "I oppose universal free tuition because opportunity and efficiency should be balanced.",
    "Public budgets are limited and must cover multiple critical sectors.",
    "A mixed model with means-tested support is more practical.",
    "Cost control and accountability should come before universal expansion.",
    "Universal subsidy may transfer public funds to families that do not need support.",
    "Priority should be targeted grants, not across-the-board fee abolition."
]

topic_sentence_support_platform_regulation = [
    "Stronger platform regulation is needed to improve accountability and information integrity.",
    "Large platforms shape public discourse and should face proportionate public obligations.",
    "Transparency requirements can reduce manipulation and opaque amplification.",
    "I support tighter regulation to protect democratic communication infrastructure.",
    "Content governance rules should be clearer and consistently enforceable.",
    "Regulation can focus on process transparency rather than viewpoint control.",
    "Public oversight is necessary when private platforms hold systemic influence.",
    "Better governance standards can reduce harm while preserving open debate.",
    "Procedural transparency can improve trust in moderation and ranking systems.",
    "Accountability rules should match the systemic role these platforms now play."
]

topic_sentence_oppose_platform_regulation = [
    "Overregulation may chill legitimate speech and political criticism.",
    "Broad compliance burdens can entrench incumbents and hurt smaller entrants.",
    "Government influence over online discourse requires strict limits.",
    "I oppose stronger top-down control because implementation risks selective enforcement.",
    "Self-regulation with independent audits may be safer than heavy statutory control.",
    "Policy should avoid creating incentives for over-censorship.",
    "The line between harmful content and controversial opinion is often contested.",
    "Regulation must be narrowly scoped to avoid damaging open communication.",
    "Compliance-heavy frameworks can unintentionally suppress grassroots political speech.",
    "Policy should minimize state leverage over lawful but unpopular viewpoints."
]

topic_to_sentences = {
    debate_topics[0]: {"support": topic_sentence_support_tax, "oppose": topic_sentence_oppose_tax},
    debate_topics[1]: {"support": topic_sentence_support_immigration, "oppose": topic_sentence_oppose_immigration},
    debate_topics[2]: {"support": topic_sentence_support_energy, "oppose": topic_sentence_oppose_energy},
    debate_topics[3]: {"support": topic_sentence_support_article9, "oppose": topic_sentence_oppose_article9},
    debate_topics[4]: {"support": topic_sentence_support_tuition, "oppose": topic_sentence_oppose_tuition},
    debate_topics[5]: {"support": topic_sentence_support_platform_regulation, "oppose": topic_sentence_oppose_platform_regulation},
}
