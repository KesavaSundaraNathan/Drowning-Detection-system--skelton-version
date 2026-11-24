# Owner steps: seed check, test-set labelling, adding v0 to the app

Three jobs, in this order. Do Part A first: it unlocks v0 training. Part B can be done in any free hours, before or
after v0. Part C comes after v0 has finished and you have read its report.

---

## Before you start (every session)

1. Start **Docker Desktop** and wait until it says "Engine running". CVAT then starts by itself; give it ~1 min.
2. Open **http://localhost:8080** in your browser (only works on this PC).
3. Log in as `lifeguard_admin`. The password is in `C:\lifeguard\training\cvat\credentials.txt`.
4. Open **Projects → Life Guard swimmers v1**. All the tasks below are in this project.

If the page doesn't load after 2 minutes, open Docker Desktop → Containers and check that `cvat_server` and
`cvat_ui` are running (green). Restart them there if needed.

### CVAT keys you will use all the time

| Key | Does |
|---|---|
| **F** / **D** | next / previous frame |
| **Tab** / **Shift+Tab** | next / previous box on this frame (Attribute annotation mode; wraps round, does not change frame) |
| **0**–**3** | pick the `verdict` of the zoomed box (Attribute annotation mode) |
| **N** | draw a new shape (repeats the last tool and label) |
| **Del** | delete the selected box |
| **Ctrl+S** | save (do this often) |
| **F1** | the full shortcut list |

### Marking a job finished

When a job is done: save (**Ctrl+S**), then open **Menu** (top left of the job) → **Finish the job** →
**Continue** (it saves and sets the job state to completed). That state is how the system knows a job is done: you
don't need to send any message.

---

## Part A — CHECK the seed boxes, one key per box (tasks 7 and 8, ≈ 2–3 h in total)

What it is: 250 frames, but **you don't draw or edit**. Every box the detectors proposed is already there as a pink
`proposal` box. You only answer, box by box, "what is this?". v0 trains on your answers.

Jobs: **task 7 "CHECK R1 train"** = jobs 29–35 (30 frames each); **task 8 "CHECK R1 val"** = jobs 36–37 (25 each).
(Tasks 5 and 6, now named "OLD … superseded", are no longer used: ignore them.)

**Do one val job (36 or 37) early**: v0 cannot start without at least one finished val job.

### 1. Answer every box (Attribute annotation mode)

1. Open a job. At the top right, switch the mode from **Standard** to **Attribute annotation**.
2. CVAT zooms onto one box at a time. Its `verdict` choices have number keys **starting at 0**:

   | Key | verdict | Use when |
   |---|---|---|
   | **0** | `?` | (not decided: never leave a box on this) |
   | **1** | `swimmer` | any part of the body is in the water |
   | **2** | `person` | completely out of the water (deck, lifeguard, spectators) |
   | **3** | `not_a_person` | reflection, poster, float, splash, empty water, or a second box on someone already boxed |

   The panel on the right shows the same list as "[0]: ? [1]: swimmer [2]: person [3]: not_a_person".
3. **Boxes the detectors agreed on already show `swimmer` or `person`.** If that's right, just press **Tab** for the
   next box. Press a key only when it's wrong.
4. **Boxes showing `?` need a key**: 1, 2 or 3, then **Tab**.
5. The counter at the top of the panel ("proposal 2 [2/14]") shows where you are. After the last box, Tab goes back
   to the first box of the same frame, so press **F** for the next frame. **Ctrl+S** every few frames.

A box that is a little loose is fine: answer what it is, don't adjust it.

### 2. Add anyone still missing (Standard mode, ~10 s per frame)

1. Switch back to **Standard** mode and step through the job's frames (**F**).
2. If you see a person with **no box at all**, press **N**, pick `swimmer` or `person`, shape **Rectangle**, and click
   two opposite corners. Every visible head gets a box, even a head alone.
3. **Ctrl+S**, then **Finish the job** (see above).

A job with any box still on `?` is left out of training and listed in `training\reports\label_qa_R1.md`, so nothing
is silently missed.

### When to stop

- When **8 of the 9 jobs** are finished (with at least one val job among them), training starts by itself within
  5 minutes and you get a desktop alert. Keep the PC on: ≈ 6.5 h, ending with another alert and the report
  `training\reports\train_v0.md`.
- The 9th job can be finished later: it joins the next retrain.

---

## Part B — Label the TEST set (≈ 3–4 h; 150 single frames + the dense segment)

What it is: the frames the final score is measured on. Nothing is pre-labelled here, on purpose: if a model wrote the
test labels, the final score would only measure how well v0 copies that model. Your shortcuts are SAM click-to-box
and SAM 2 following the people you boxed. These frames are never used for training.

**Only these jobs are needed:** TEST C jobs **14–19**, TEST B jobs **6–9**, TEST A jobs **2–3**.
(Leave TEST B jobs 10–13 and TEST A jobs 4–5 untouched: they are no longer required.)

### 1. TEST C — the 10-second dense segment (jobs 14–19): box once, SAM 2 follows

1. Open **job 14**. On its **first frame**, draw a plain box around **every person** (**N** → `swimmer` / `person` →
   shape **Rectangle** → keep **Shape**, not Track). **Ctrl+S**, then close the job tab.
2. Double-click **`C:\lifeguard\training\tools\follow_test_c.cmd`**. It takes 1–2 minutes and ends with
   "wrote N tracks". (Training running at the same time is fine.)
3. Reopen job 14: each person is now a **track** running through all 251 frames. Step through with **F**:
   - a box slipped off its person → drag it back on that frame;
   - the person is hidden or has left the picture → select the track and press **O** ("outside") on that frame;
   - it switched to another person → fix the box on that frame and the frames after it.
4. **Someone appears later in the segment?** On the first frame you see them, draw a plain box (as in step 1), save,
   close the tab, and run `follow_test_c.cmd` again. Only the new boxes are followed; finished tracks stay as they are.
5. Check jobs 15–19 the same way (they show the same tracks), then **Finish** jobs 14–19.

### 2. TEST B — jobs 6–9 (boxes + class, 25 frames each, ≈ 15–20 min per job)

For every frame: a **Rectangle** around every visible person, label `swimmer` or `person` (same rule as Part A).
Shortcut: left toolbar → **AI Tools** → **Interactors** → **Segment Anything** → click the person → **Enter**; if it
gives an outline instead of a box, draw the rectangle over it (**N**) and delete the outline (**Del**).
**Ctrl+S**, **F**, and **Finish** each job.

### 3. TEST A — jobs 2–3 (boxes + class + body keypoints)

1. Boxes and classes exactly as in TEST B.
2. For every person **at least ~40 px tall**, add the **`pose`** skeleton (17 points: nose, eyes, ears, shoulders,
   elbows, wrists, hips, knees, ankles) and drag each point onto its joint:
   - joint clearly visible → leave it as a normal point;
   - under water / refracted but you can place it confidently → right-click the point → **occluded**;
   - you cannot place it (hidden, out of frame, lost in splash) → right-click → **outside**.
   **Never guess a joint.** Left / right are the person's own left and right.
3. **Ctrl+S**, **Finish** each job.

Detailed rules for every case (overlaps, splash, glare, cut-off people): `training\docs\LABELING_GUIDE.md`, sections
1–5 and 7.

When all the jobs above are finished, tell Claude: the test frames are exported and the final evaluation is run once
(`training\tools\eval_v0.py --final`).

---

## Part C — Add v0 to the app's model list (after v0 finishes, ~30 min)

Only do this if the report says v0 is worth trying. The app is untouched until you do.

### 1. Read the report and write down your rule first

Open `C:\lifeguard\training\reports\train_v0.md`. It compares `dfine_l_v0` and `rtmdet_l_v0` with the app's current
detectors on the checked val frames (AP50, recall at 0.1 / 0.5 / 1 false positive per frame, small-person recall,
swimmer-vs-person accuracy). Both v0 files must show **Life Guard decode check PASS**.

Before any app evaluation, **write your pass rule down** (for example "recall at 0.5 FP per frame at least 5 points
better than RTMDet-m, and no more false alerts on the test split"). The project's rule is: tune on dev, run test
once, rule written before the run.

### 2. Copy the model into the app

PowerShell, from `C:\lifeguard`:

```powershell
Copy-Item training\work_dirs\onnx\dfine_l_v0.onnx backend\models\lifeguard_dfine_l_pool_v0.onnx
Get-FileHash backend\models\lifeguard_dfine_l_pool_v0.onnx -Algorithm SHA256
(Get-Item backend\models\lifeguard_dfine_l_pool_v0.onnx).Length
```

(Use `rtmdet_l_v0.onnx` and an `lifeguard_rtmdet_l_pool_v0.onnx` name instead if RTMDet is the better one.)

### 3. Register it in `backend\models\MODELS.json`

Add one entry to the `"models"` list (copy the `lifeguard_dfine_m_devtrain_v1_dynbatch.onnx` entry and change it):

```json
{
 "file": "lifeguard_dfine_l_pool_v0.onnx",
 "role": "detector",
 "key": "lifeguard_dfine_l_pool_v0",
 "kind": "det",
 "source": {
  "framework": "D-FINE (Apache-2.0)",
  "base_checkpoint": "dfine_l_coco.pth",
  "config": "training/configs/dfine_l_v0.yml",
  "trained_checkpoint": "training/work_dirs/dfine_l_v0/best_stg2.pth"
 },
 "sha256": "<the hash from step 2, lower case>",
 "bytes": <the length from step 2>,
 "licence": "Apache-2.0 (D-FINE code)",
 "commercial_ok": true,
 "data_provenance": "D-FINE-L COCO checkpoint fine-tuned on the owner-checked seed (250 frames) of this pool's own footage, 2026-10",
 "classes": ["swimmer", "person_out_of_water"],
 "dev_only": true
}
```

- Keep `"dev_only": true` until the footage's rights are cleared for shipping (the client's call). A dev-only model
  can still be chosen per deployment, never as the default.
- `trained_checkpoint`: use the file name the report shows (it may be `best_stg1.pth`).
- For RTMDet: `"framework": "MMDetection RTMDet (Apache-2.0)"`, `"licence": "Apache-2.0 (MMDetection code)"`,
  `"base_checkpoint": "rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"`.

### 4. Check it

```powershell
$env:PYTHONPATH = "backend"
venv\Scripts\python tools\verify_models.py
venv\Scripts\python tools\check_licenses.py
```

Both must pass (the hash and size must match the file).

### 5. Try it on uploaded clips first

Set it as the upload detector only, restart the upload worker, and upload a few pool clips on the Uploads page:

```powershell
$env:LIFEGUARD_UPLOAD_DETECTOR_MODEL = "C:\lifeguard\backend\models\lifeguard_dfine_l_pool_v0.onnx"
venv\Scripts\python -m lifeguard.jobs.upload_worker
```

Look at the Model View: one box per swimmer, swimmers vs deck people right, nothing on reflections.

### 6. Measure it the project's way

```powershell
venv\Scripts\python -m lifeguard.eval.evaluate --eval-dir eval\heldout_v1 --split dev --mode High --provider tensorrt --name v0dev --out build\v0_eval --det-model backend\models\lifeguard_dfine_l_pool_v0.onnx
```

Then, with your rule written down, run `--split test` **once**. Put the commands and numbers in
`docs\evidence\<phase>\`.

### 7. Live cameras (only after step 6 passes)

Point the live detector at the new file (set this in the terminal that starts the API), then rebuild the GPU
engines and restart the API:

```powershell
$env:LIFEGUARD_DET_MODEL = "C:\lifeguard\backend\models\lifeguard_dfine_l_pool_v0.onnx"
Remove-Item C:\LifeGuardDev\capacity_cache.json
venv\Scripts\python tools\build_trt_engines.py --data-dir C:\LifeGuardDev --cameras 4 --fps 18
```

The licence gate keeps `dev_only` models from being the shipped default; if the app refuses the file for live use,
live use waits until the footage rights are cleared and `dev_only` is set to `false`. The live APIs stay off until all
training and testing are finished (your instruction of 2026-10-28).

### 8. Commit

Commit only `backend\models\MODELS.json` (and the evidence notes). The `.onnx` file itself never goes into git.
