# > data / [ breach_evidence_locker ]
### *"the system saw everything. we just taught it what to look for."*

---

All datasets live **on my lappy** — raw surveillance footage is too heavy for GitHub and too precious for compression artifacts.  
If you're cloning this repo and need to replicate results: download links below, drop files into the matching folder, and you're surveillance-ready.

> ⚠️ `data/raw/` is gitignored. you won't find the footage here. follow the instructions. atp everyone should just listen to me😏

---

## 📁 folder map

```
data/
├── raw/
│   ├── abandoned_objects/
│   │   ├── ABODA-master/         ← primary eval
│   │   └── AVSS 2007/            ← stress-test (replaces PETS2006)
│   │
│   ├── unauthorised_access/
│   │   ├── CDnet/                ← primary eval
│   │   └── VIRAT/                ← stress-test
│   │
│   └── suspicious_activities/
│       ├── UFC_Crime/            ← primary eval
│       └── Avenue Dataset/       ← stress-test
│
└── sample_videos/                ← small clips for quick inference sanity checks
```

---

## 🗂️ dataset registry

### [ MODULE 01 ] — abandoned object detection

| role | dataset | what it is | download |
|------|---------|------------|----------|
| 🔴 primary | **ABODA** | 11 `.avi` sequences — crowded scenes, occlusions, lighting chaos. the messy real world. | [github ↗](<https://github.com/kevinlin311tw/ABODA>) |
| 🟡 stress-test | **AVSS 2007 / i-LIDS** | UK Home Office standard. underground station. bags left unattended. the OG benchmark. picked it up from someone else's project | [mirror ↗](<http://www-vpu.eps.uam.es/publications/AODsurvey/>) |

> used for: validating the `ATTENDED → WARNING → ALARM` state machine + owner-object proximity logic

---

### [ MODULE 02 ] — unauthorised access / zone intrusion

| role | dataset | what it is | download |
|------|---------|------------|----------|
| 🔴 primary | **CDnet 2014** | `intermittentObjectMotion` + `PTZ` categories only. pixel-wise ground truth. classic baseline comparator. | [kaggle ↗](<https://www.kaggle.com/datasets/vafaeii/cdnet-2014-change-detection-benchmark-dataset>) |
| 🟡 stress-test | **VIRAT Ground 2.0** | `videos-01.zip` + annotations. outdoor HD surveillance. real scenes, real clutter. | [kitware ↗](<https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html>) |

> used for: testing zone-crossing detection logic across controlled and uncontrolled environments

---

### [ MODULE 03 ] — suspicious activity recognition

| role | dataset | what it is | download |
|------|---------|------------|----------|
| 🔴 primary | **UCF-Crime** | 1900 real-world CCTV videos. categories used: Burglary, Robbery, Fighting, Stealing, Vandalism. | [UCF ↗](<https://www.dropbox.com/scl/fo/2aczdnx37hxvcfdo4rq4q/AOjRokSTaiKxXmgUyqdcI6k?e=4&from_auth=register&ignore_new_user_install_redirect=1&rlkey=5bg7mxxbq46t7aujfch46dlvz&dl=0>) |
| 🟡 stress-test | **CUHK Avenue** | 16 train + 21 test videos. campus CCTV. 47 annotated abnormal events. includes ground truth. | [CUHK ↗](<https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html>) |

> used for: validating suspicious behaviour flagging across crime-heavy and everyday-anomaly scenarios

---

## 🧠 usage philosophy

these datasets are **not used for training.**  
YOLO already knows what a person, a bag, a body in motion looks like. [clear your basics already🥀]

what we're testing is the **engineering layer on top** —  
the state machine. the proximity logic. the timer thresholds. the memory that survives an occlusion.

[its my brain baby i promise😉]
```
baseline_eval.py  →  raw YOLO + tracker, no state machine  →  results/baseline/
custom_eval.py    →  full pipeline with engineering layer  →  results/custom/
```

same input. same footage. two pipelines. one comparison table.  
that's the paper.

---

## 📐 evaluation split logic

| dataset type | what it proves |
|---|---|
| primary dataset | main benchmark numbers — Precision, Recall, FPS |
| stress-test dataset | generalisability — does the system hold up in a different environment? |

AGAIN-
this is not a train/test split in the ML sense.  
it's a **primary evaluation / stress-test** split.  
the distinction matters. cite it correctly.   
Don't fail me, son.

---

*// I-got-my-Eyes-on-You · data layer · sweet summer of '26 ♡*