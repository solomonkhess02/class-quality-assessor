# Write-up (draft, max 300 words)

**Approach.** Code measures what it can: talk statistics, warm-up length, face visibility, lighting and connectivity incidents. An LLM (DeepSeek) scores a rubric built from the organiser's feedback and must quote the transcript. Code checks every quote, and a fixed written rule turns the findings into the label. Ten labelled classes are too few to train on; a written rule keeps labels explainable. Audio and video stay on the server; the LLM sees only pseudonymised text and numbers.

**What worked.** On the 10 labelled classes:
- pairwise ordering 83% (chance 50%);
- within one level 9.7/10 (always "Average" gets 8/10);
- exact label 4/10, no better than always "Poor" (4/10);
- 99% of quotes verified;
- identical labels across 3 runs for 9 classes;
- 87% of the organiser's points visible in the transcript caught;
- both input paths pass the privacy audit.

**What failed.**
- LLM scores bunched at 3. A weighted mean never produced Poor, and "energy and warmth" got 3 almost everywhere (E10).
- The LLM rated technical quality 2 for nearly every class, so code scores it now (D14).
- A guard against false "no rapport" findings cleared them all (nearly every class has some greeting); Poor vanished until rapport became warm-up time (E17, E18).
- LLM tutor/student tagging of whisper output barely beat "everyone is the tutor": 65-72% vs 55-64% (E21).
- Whisper ran 2-3x slower than a test clip suggested, so only parts of each class are transcribed (E21, E22).
- Numeric confidence was higher on wrong labels; the report dropped it.

**Limitations.**
- Voice energy and the 8x8 background are not measured.
- The face check is unvalidated: the screen-share tag matched hand labels on 74% of frames.
- The same 10 classes shaped the rules, so the numbers are optimistic, and the rule is harsh (bias -0.43).
- Without a VTT, speakers are unknown and labels are less reliable.

---

## Key moments (AI-log cover page)
- **E1:** organiser feedback became the rubric. Half of it is visible only on video, so a transcript-only system was ruled out.
- **D4, E11, D16:** DeepSeek receives pseudonymised text only. A leak audit with a human-reviewed band gates every release.
- **E9, D10:** extract-then-score lost to direct scoring. LLM counts were unreliable, while verified yes/no facts (homework, greeting by name) told classes apart.
- **E10, D14, D15:** a weighted mean could not produce Poor. The constant technical problem moved to code, and the near-constant communication score was left out of the label.
- **E16, D18:** the severity rule was written from the organiser's wording before the run and kept by a pre-registered rule (pairwise 75% to 84%).
- **E17, D20:** measurement fixes were kept although the keep rule rejected them. A bug is not kept because it scores better.
- **E18, E20, D21:** rapport became warm-up time before the first content line. The final configuration reached 88% pairwise, and accuracy work was frozen.
- **E21-E23:** the no-VTT path. Whisper speed forced shorter windows, speaker tagging failed, and labels agreed with the VTT path on 0 of 2 classes.
