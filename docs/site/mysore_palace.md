# Mysore Palace during Dasara: site data

Collected 2026-10-05 from online sources only. Nothing here was observed by us
(CLAUDE.md: no new data collection). Every number is tagged:

- **PUB**: published, with its source;
- **LIT**: a standard value from the crowd-flow literature;
- **DER**: derived by us from PUB/LIT numbers (the arithmetic is shown);
- **GAP**: not published; the scene must treat it as a stated assumption and sweep a range.

## Project framing

**Title:** A real-time digital twin of Mysore Palace crowds during Dasara.

**Problem.** Every Dasara the palace takes its largest crowds:
- up to 21,578 ticketed visitors in a day (2024);
- on Vijayadashami, a seated audience whose size changed three years running: 38,000, then 59,600, then 48,000;
- over 70,000 people in front of the palace in 2024;
- a 2025 seat cut that followed the Bengaluru stampede.

These decisions are made without a tool that shows how tens of thousands of people move through
a 19th-century compound's gates and passages:
- offline crowd tools are not interactive;
- game engines keep the frame rate by simplifying people far from the camera, so the jam nobody
  is looking at is the one they misreport.

We build an explorable 3D twin of the palace grounds that runs in real time and keeps every
visitor's behaviour faithful, seen or unseen.

| | |
|---|---|
| **Goal** | SDG 11, Sustainable Cities and Communities. 11.4: safeguard cultural heritage. 11.7: safe, inclusive public spaces for older persons and persons with disabilities. |
| **CG & VR** | Compound modelled from open map data; real-time rendering of tens of thousands of people with level of detail; crowding heat map *(planned)*; 100,000-bulb illumination *(planned)*; first-person walkthrough (desktop; a headset needs OpenXR, a new dependency). |
| **PA & GPU** | Routing of every visitor to the gates, CPU reference first and a GPU kernel later *(planned)*; parallel crowd step; per-frame detail allocation within a measured frame-time budget; AI-model inference sharing the GPU with rendering. |
| **Outcome** | Planners compare seat counts, gate openings and staggered release before the festival. Measures: exit time against the 8-minute guide; flow per gate against 82 persons per metre per minute; peak crowding by Fruin level; waiting time of older visitors. |

**Why it is not stereotypical:** a real site and a decision documented three years running. It
covers routine festival operations, not a generic evacuation.

**Patent:** the patent covers the system and method, and the palace is one application. The
slides say what the method achieves, not how the allocator works inside.

## What the study is for

**1. Testing plans before the day.** The 8-minute rule gives the exit width needed if everyone
spreads evenly over the gates (section 5). The simulation shows where that breaks down:
- people head for the nearest gate;
- families move together, and older visitors walk slower;
- the procession leaves through the Balarama gate just as the seated crowd stands up.

Questions it answers:
- Does releasing enclosures in waves clear the grounds faster?
- What does one more open gate buy?
- Should each enclosure be sent to its own gate?
- Where does crowding peak? That is where to station police and ambulances.
- How many ticket counters serve about 2,900 visitors an hour?

Crowd analysis has changed real sites before. The 2006 Hajj crush was analysed from video
(Helbing, Johansson and Al-Abideen, *Phys. Rev. E* 75, 046109, 2007), and the Jamarat Bridge
was rebuilt afterwards. UK stadium capacities follow the Green Guide's exit-flow rules used here.

**Limit.** There are no measured counts from the palace, so the results are **comparative, not
predictive**:
- defensible: "plan B clears 30% faster than plan A under the same assumptions";
- not defensible: "48,000 is safe".

A real safety decision needs the Palace Board's gate widths and counts. With them, the same tool
becomes calibrated.

**2. Why it needs this method.** A real-time 3D twin is useful because a planner can move around
and watch. Distance-based level of detail makes the numbers depend on where the camera is.
Measured on our scenes (STATE.md, figure 2):
- the distance-band baseline shifted trip times by up to 59 s across five camera paths;
- PARITY gave identical results on all five.

At the palace, the selling point is that the answer does not change when you move the camera.
Feeding the twin live gate counts is future work, not a claim.

## The system and results so far

PARITY decides, every frame, how much detail each person in the crowd gets on four axes:
behaviour, navigation, animation and geometry. It works under two budgets at once:
- the measured frame time;
- a per-person cap on how far behaviour may drift from a full-detail reference, which forces
  periodic restoration.

The baseline it is compared with is Unreal Engine 5's MassLOD: distance bands and fixed caps per
level. Detail of the latest method work stays off the slides until the IPR cell has advised.

Measured so far, before the palace (full log in README.md and STATE.md):

| Question | Result |
|---|---|
| Is drift bounded? | PARITY's worst agent stays at or under its cap in every test; the baselines reach 3.3 to 6 times the cap. |
| Do results depend on the camera? | PARITY: identical results on all five camera paths. Distance-band baseline: trip times shift by up to 59 s. |
| LLM-driven agents, same model budget | Drift 2 to 4.8 times lower than view-first or round-robin scheduling; worst agent 0.94 to 0.98 of the cap, theirs 1.2 to 2.4 times it. |
| Rendering 8,000 agents (RTX 2050 laptop, 3 seeds) | Against the strongest baseline (a full-simulation render knapsack) at equal frame time: at 20 to 30 ms better perceived quality, the same image quality and 3 to 10 times fewer visible pops. At 13 to 16 ms it was behind. A fix is verified on a Mac; the laptop rerun is pending. |
| Correctness | The C# engine port agrees with the Python reference on every allocator decision. |
| Negative results, kept | Separate CPU and GPU budgets gave no gain on CPU-bound frames; a 1.5B model as a middle tier added nothing. |

Courses: CGVR (CS373IA) and Parallel Architecture and GPU Programming (CS372IA). Built in Unity 6
(rendering, engine port), Python (reference simulator and oracle), and CUDA/numba (parallel allocator).

## 1. Dates (2026)

| Fact | Value | Tag |
|---|---|---|
| Mysuru Dasara | Sun 11 Oct to Wed 21 Oct 2026 | PUB [1][2] |
| Jamboo Savari (Vijayadashami) | Wed 21 Oct; Mysuru follows its own panchanga, so a day after the national 20 Oct | PUB [1] |
| Palace opening hours | 10:00 to 17:30 daily, except the last two days of Navaratri | PUB [3] |
| Illumination | about 100,000 bulbs (15 W incandescent), every evening; hours quoted as 19:00 to 21:00 or to 22:00 | PUB [3][9] |

## 2. Site geometry

Local frame for the scene: origin 12.3050 N, 76.6560 E, x east and y north, in metres.
`mysore_palace_osm.json` is the raw OpenStreetMap extract (API 0.6 `map` call, bbox
76.6520,12.3022 to 76.6600,12.3082, fetched 2026-10-05). Licence ODbL: credit
"© OpenStreetMap contributors"; anything derived from it inherits ODbL.

| Fact | Value | Tag |
|---|---|---|
| Palace compound (OSM way 656323590) | 239,047 m² = 59.1 acres, perimeter 2.52 km, extent x -350..366, y -233..289 | DER from OSM [4] |
| Compound area, secondary sources | 72 acres (the OSM outline may leave out parts of the old fort grounds) | PUB [5] |
| Gardens and grass inside the compound | about 119,000 m² (OSM `leisure=garden/park`, `landuse=grass`) | DER from OSM [4] |
| Main palace block | 75 m wide x 48 m deep x 44 m high, 3 floors, granite | PUB [3] |
| Palace footprint with its courtyards (OSM relation 5367659) | 15,283 m², x -294..-109, y -28..93; it faces east onto the forecourt | DER from OSM [4] |
| Public Durbar Hall (Diwan-e-Aam), first floor | 47 m x 13 m | PUB [3] |
| Gates | Jaya Marthanda (east), Jayarama and Balarama (north), Varaha (south), Brahmagiri (west) | PUB [3] |
| Public entry | Varaha (south) gate; the OSM entrance node is at (-75, -169), with "Vahara ticket counter" at (-90, -172) | PUB [6], OSM [4] |
| East gate | OSM: "Jayamarthanda Gate (Palace Main Gate)", `access=no`, at (199, -4); a secondary source says it opens in Dasara only for dignitaries | OSM [4], PUB [6] |
| Procession start | Balarama gate (north) | PUB [3] |
| Other mapped entrances inside the compound | (-317, -114), (-295, -18), (-31, 173), (-326, 146) | OSM [4] |
| Gate clear widths | not published, and OSM maps gates as points | **GAP** |

## 3. Crowd numbers

**Ticketed daytime visitors, Dasara 2024** (Mysore Palace Board figures via [7]). The palace was
closed 3 Oct 10:00 to 14:00, and all day on 11 Oct (Ayudha Puja) and 12 Oct (Vijayadashami).

| Date (2024) | 3 Oct | 4 Oct | 5 Oct | 6 Oct | 7 Oct | 8 Oct | 9 Oct | 10 Oct | Total |
|---|---|---|---|---|---|---|---|---|---|
| Visitors | 7,129 | 12,247 | 18,307 | **21,578** | 13,931 | 12,067 | 13,568 | 13,243 | 112,070 (676 foreigners) |

Earlier years: 121,612 in eight days (2023, 756 foreigners) and 67,147 (2022, 260 foreigners) [7].
The palace has about 4 million visits a year (2023-24) [3].

**Jamboo Savari seating on the palace premises:**

| Year | Seats | Note | Tag |
|---|---|---|---|
| 2023 | 38,000 | | PUB [8] |
| 2024 | 59,600 | "Over 70,000 people watched Dasara procession in front of the Palace" (Deputy Commissioner); a gold-pass holder reported more passes issued than seats and trouble getting in | PUB [7][8] |
| 2025 | 48,000 | cut by 11,600 after the 4 June 2025 stampede near Chinnaswamy Stadium, Bengaluru (11 dead); police crowd SOP applied | PUB [8] |
| 2026 | not yet published | | **GAP** |

Entry on Jamboo Savari day is by pass, "gold card" or ticket [8]. Over 5 lakh people watched along
the procession route in 2024 [7]. In 2025 the police planned evacuation routes and put
ambulances and fire vehicles at the palace gates [10].

**Illumination evenings:** no published head count. Guides only say "thousands" gather. **GAP**

## 4. Reference values from the literature

| Quantity | Value | Tag |
|---|---|---|
| Mean walking speed, Indian pedestrians, 18 sites in 5 cities | 71.22 m/min (1.19 m/s) on sidewalks, 63.60 m/min (1.06 m/s) in pedestrian precincts | LIT [11] |
| Exit flow, level surface (UK Green Guide) | 82 persons per metre width per minute (1.37 p/m/s) | LIT [12] |
| Exit flow, stepped surface | 66 persons per metre width per minute (1.10 p/m/s) | LIT [12] |
| Normal egress time used for exit capacity | 8 minutes | LIT [13] |
| Fruin walkway levels of service (space per person) | A ≥ 3.25 m², B 2.32-3.25, C 1.39-2.32, D 0.93-1.39, E 0.46-0.93, F < 0.46 | LIT (Fruin 1971, *Pedestrian Planning and Design*) |

## 5. First-order check

Total exit width needed to clear the Jamboo Savari seats in 8 minutes: seats / (rate x 8).
This assumes everyone uses the exits evenly. That is exactly the assumption the simulation
tests: people head for the nearest gate, move as families, and meet the procession's own
movement.

| Seats | Level exits (82/m/min) | Stepped exits (66/m/min) | Tag |
|---|---|---|---|
| 38,000 (2023) | 58 m | 72 m | DER |
| 48,000 (2025) | 73 m | 91 m | DER |
| 59,600 (2024) | 91 m | 113 m | DER |
| 70,000 (2024 reported crowd) | 107 m | 133 m | DER |

Daytime visits on the busiest 2024 day: 21,578 over 7.5 opening hours is 2,877 per hour, or 48 per
minute on average through the Varaha gate and ticket counter. **DER.** The peak-hour factor is a **GAP**.

## 5b. Gate audit (2026-10-05)

Positions come from OSM, where mapped paths cross the compound outline. Roles come from news
reports. Widths are estimated from Wikimedia Commons photographs, scaled by a person or a scooter
in frame or by the gate structure's mapped footprint. These are **estimates (DER)**: no source
publishes gate widths. Esri World Imagery was used only to check each gate's setting; no imagery
is reproduced.

| Model key | Gate | Role on Jamboo Savari day | Clear width estimate (range) | Evidence |
|---|---|---|---|---|
| `east` | Jayamarthanda, the main gate | Police staffed [U][DH20]; elephants enter by it [FPJ]; VIP and pass entry | **10 m** central arch (8-12), plus two ground arches of about 3 m | Structure 34 m along the wall, 15 m deep, towers 17 m (OSM); arch is 0.32 of the front width in a frontal photo [C1] |
| `south` | Varaha, the public entry | Police staffed [U][DH20]; daily ticket entry | **4.2 m** central arch (3.5-5), plus two side doorways of about 1.1 m | People in the arch for scale [C2]; temple gopuram inside matches the Shweta Varahaswamy temple (OSM) |
| `north` | Jayarama and Balarama, two separate arches side by side | The procession leaves by Balarama [W]; police staffed [DH20] | Per arch **4.7 m** central (4-5.5) plus two side arches of about 1.9 m; both arches about 17 m | Scooters in frame for scale [C3]; two structures [C4] |
| `west_n` | Brahmapuri (*which west opening is which is not confirmed*) | Police staffed [U] | **4 m** ASSUMED (3-6) | West gate used by the Parakala Mutt pontiff [V]; mapped structure 22 x 10 m (OSM) |
| `west_s` | Karikal Thotti (*not confirmed*) | Police staffed [U][DH20] | **3 m** ASSUMED (2-4) | On the west side [KT]; OSM `access=private` (daily); structure 14 x 3 m |
| `east_s`, `east_n` | Side openings into the walled forecourt outside Jayamarthanda | Not in any report | 3 m ASSUMED | OSM pedestrian crossings; satellite shows they open into the same outer forecourt as the main gate |

What the gates look like on the outside, from satellite and OSM:
- **Varaha:** opens onto a paved approach about 35 x 80 m, then the main road.
- **Jayamarthanda:** opens into a walled outer forecourt about 100 m across, with an avenue of about
  20 m to the main road.
- **North gates:** face a large parking plaza and a roundabout.
- **West gates:** open onto the road on that side; parking lies south of the southern one.

**Entry is by pass only, through door-frame metal detectors after frisking** [DH20], so on arrival
the bottleneck is screening, not width. Industry figures for one walk-through lane are 600-1,200
persons per hour, with a stadium tender asking at least 1,100 [2M][PB]. Full frisking is slower,
so the model uses 600 (sweep 300-1,200). The number of lanes per gate is not published:
ASSUMED one lane per 1.5 m of clear width.

Gate sources:
- [U] Udayavani, "Jamboo Savari: security beefed up around Mysuru Palace":
  https://english.udayavani.com/article/jamboo-savari-security-beefed-up-around-mysuru-palace/937608
- [DH20] Deccan Herald, "Vijayadashami procession amidst tight security":
  https://www.deccanherald.com/india/karnataka/vijayadashami-procession-amidst-tight-security-907476.html
- [FPJ] Free Press Journal, Mysuru Dasara 2026 elephant squad:
  https://www.freepressjournal.in/lifestyle/mysuru-dasara-2026-majestic-elephant-squad-receives-royal-welcome-at-the-iconic-palace-marking-indias-one-of-the-celebrated-festivals
- [W] Wikipedia, "Mysore Palace".
- [V] vayusutha.in, Brahmapuri gate Anjaneya temple: https://vayusutha.in/vs4/temple198.html
- [KT] Search summary: Karikal Thotti and Brahmapuri gates on the west side.
- Wikimedia Commons, Category:Gates of the Mysore Palace:
  - [C1] "Mysore palace gate.jpg" (CC BY-SA 4.0);
  - [C2] "MysoreMaharajaPalaceEntrance.jpg", Ravikiran Kaja (CC BY-SA 2.5);
  - [C3] "Mysuru Palace - North Gates - 3.jpg" and [C4] "- 2.jpg" (CC BY-SA 4.0).
- [2M] 2M Technology, event screening queue optimisation:
  https://www.2mtechnology.net/event-screening-queue-optimization/
- [PB] PeoplesBank Arena metal detector RFP answers:
  https://www.peoplesbankarena.com/assets/doc/PeoplesBank-Arena-Response-to-Questions-Metal-Detector-RFP-41177e6c70.pdf

## 6. Gaps the scene must state as assumptions

- Clear width of each gate, and which gates are open for entry and for exit on Jamboo Savari day.
- Layout of the seating enclosures in the forecourt.
- Arrival profile: when pass-holders arrive, and the daytime peak-hour factor.
- Group sizes (families) and the share of elderly visitors. Take the elderly share from the Census of India 2011.
- Crowd size on illumination evenings.
- Interior visitor route and corridor widths. Photography inside is restricted, so the interior is approximate.

## 7. Scenarios this supports

1. **Jamboo Savari egress (headline).** Seats 38,000 / 48,000 / 59,600 / 70,000 crossed with
   gate-width assumptions. Report egress time and peak density at each gate against the 8-minute
   rule and Fruin levels. The sudden shift: the howdah leaves through the Balarama gate and tens
   of thousands stand up and move at once.
2. **Peak-day ticketed visit.** 2,877 per hour average through the Varaha gate, ticket counter and
   footwear deposit, then the one-way route through the palace. These are the queues and narrow
   passages.
3. **Illumination evening.** At switch-on the crowd stops and turns to the palace; at switch-off it
   heads for the gates. There is no published count, so sweep the crowd size.

## Sources

1. Siasat, "Mysuru Dasara 2026 from October 11, Jumbo Savari on October 21": https://www.siasat.com/mysuru-dasara-2026-from-october-11-jumbo-savari-on-october-21-3505227/
2. Hindutone, "Mysore Dasara 2026: day-by-day guide": https://hindutone.com/festivals/mysore-dasara-2026-day-by-day-guide/
3. Wikipedia, "Mysore Palace": https://en.wikipedia.org/wiki/Mysore_Palace (fetched 2026-10-05)
4. OpenStreetMap contributors, extract in `mysore_palace_osm.json` (ODbL): https://www.openstreetmap.org/way/656323590
5. Housing.com, "Mysore Palace": https://housing.com/news/mysore-palace ; GoAsia guide: https://goasia.cc/en/article/mysore-palace
6. Ancient History Sites, "Mysore Palace": https://www.ancient-history-sites.com/?p=123314
7. Deccan Herald, 13 Oct 2024, "Tourists throng heritage city Mysuru during ten days of Dasara": https://deccanherald.com/india/karnataka/tourists-throng-heritage-city-mysuru-during-ten-days-of-dasara-3230827
8. Deccan Herald, 19 Aug 2025, "Stampede effect: seats at Mysuru palace reduced for Dasara Jamboo Savari": https://www.deccanherald.com/india/karnataka/mysuru/stampede-effect-seats-at-mysuru-palace-reduced-for-dasara-jamboo-savari-3686450
9. Thrillophilia, "Dussehra in Mysore": https://www.thrillophilia.com/dussehra-in-mysore
10. Deccan Herald, 19 Sep 2025, "Bengaluru stampede effect: Mysuru police come up with action plan for crowd management during Dasara": https://www.deccanherald.com/india/karnataka/mysuru/bengaluru-stampede-effect-mysuru-police-come-up-with-action-plan-for-crowd-management-during-dasara-3736250
11. Rastogi, Thaniarasu and Chandra, "Design Implications of Walking Speed for Pedestrian Facilities", *J. Transportation Engineering* 137(10), 2011: https://ascelibrary.com/doi/10.1061/%28ASCE%29TE.1943-5436.0000251
12. Sports Grounds Safety Authority, "Egress" (Green Guide rates of passage): https://sgsa.org.uk/physical-factors/circulation/egress/
13. Guide to Safety at Sports Grounds (Green Guide), section 10.11, 8-minute egress: https://www.jesip.org.uk/uploads/media/incident_reports_and_inquiries/Guide%20to%20Safety%20at%20Sports%20Grounds.pdf ; SGSA worked example: https://sgsa.org.uk/wp-content/uploads/2024/02/Worked-Example-A-Football-and-Rugby.pdf

## Build plan

Main zone first: the forecourt on Jamboo Savari day. Then the south gate, ticket counter and
footwear deposit, then the palace interior.

1. **Site file.** `tools/osm_site.py` turns the OSM extract into `sim/data/mysore_palace_site.json`:
   - the compound wall, building footprints and gates;
   - gate widths and the enclosure layout, flagged as assumptions.

   The Python and Unity scenes share this file.
2. **Routing** (`sim/site.py`):
   - a 0.5 m walkability grid;
   - one distance-to-gate field per gate, computed by Dijkstra on the CPU, which is the oracle;
   - visitors walk down their gate's field, and walls block crossings;
   - ORCA between people is unchanged.

   Gate choice and family grouping live in the view-invariant core (invariant 4).
3. **Palace scene** (`sim/scenes.py`):
   - phases: arrival, seated, release, egress;
   - metrics: exit time, flow per gate, Fruin crowding, waiting time.

   Development runs at N ≤ 200.
4. **Calibration.** The palace scene's drift rates are measured by the existing reference procedure (invariant 1).
5. **Tests.** Every field reaches its gate, no agent ends inside a wall or building, everyone exits, and runs are deterministic.
6. **Checkpoint.** First exit-time numbers.
7. **Unity:**
   - a `SceneKind.Palace` that reads the same site file;
   - a C# routing port, with a test that it agrees with Python;
   - a compound built from extruded footprints and primitives.
8. **Scale check:** how many agents the forecourt carries at 16 ms. The bench was validated at 8,000 agents, but Jamboo Savari seats 48,000. If 48,000 does not fit, simulate one enclosure sector at full density and say so.

Later: a GPU routing kernel with the C# version as its oracle, the illumination and switch-on
scenario, the heat-map overlay, and the other zones.

**Assumption (changed 2026-10-05):** on Jamboo Savari day the lawns are walkable, because
48,000 seats need about 3 ha and only the lawns give that much ground; `Site(gardens_walkable=False)`
is the other reading.

**Progress (2026-10-05):**
- Step 1 done. `tools/osm_site.py` writes `sim/data/mysore_palace_site.json` (45 KB): 7 gates
  (where mapped paths cross the compound outline), 29 buildings, 108 building parts with heights
  (palace towers up to 40 m), walls and gardens.
- Step 2 done. `sim/site.py`: 0.5 m grid, 20.6 ha walkable; one field per open public gate.
  Each gate reaches 98.4% of the walkable ground. Gate passages run 6 to 9 m through gatehouses,
  because the compound outline is lined with the old fort buildings.
- Step 3 done. `sim/scenes.Palace` places blocks of 800 seats (20 rows x 40, rows facing the palace)
  on open ground, nearest the palace front first. Blocks are kept 10 m from buildings and 15 m from
  gates, and clear of the procession lane to the north gate. Up to 120 blocks fit, so even 70,000
  seats fit. Each row empties to its nearer end, and each block side follows one route to its gate.
  Gate plans:
  - **nearest:** every side takes its closest gate. At 48,000 seats only the north gate (19,200)
    and the south gate (28,800) get used.
  - **balanced:** seats per gate in proportion to gate width: north 16,000, south 12,000,
    west 8,000, the two east openings 6,000 each.
- Crowd model for the exit runs (`bench/palace.py`). Each fix below came from a check against the
  literature:
  - The simulator's soft-push model alone let an 8,000-seat crowd pass the 6 m south gate at
    590 persons/m/min, at 19 persons/m². That is impossible.
  - Adding 0.25 m bodies that may not overlap still gave 354, because a packed column kept walking
    at free speed.
  - Adding Weidmann's (1993) speed-density relation on local walking density (LIT) gives a
    saturated-gate peak of 54-57 persons/m/min and at most 5.3 persons/m². Weidmann's own maximum at
    our 1.06 m/s mean speed is 58; the Green Guide design rate is 82. **Absolute exit times are on
    the slow side, so read the results as plan-against-plan comparisons.**
  - An agent counts as out once it reaches the outer half of any open gate's apron.
  - An agent that moves less than 0.5 m in 5 s follows its gate's distance field instead of its
    next waypoint. Without this, 123 of 8,000 were pinned against a temple wall after the crowd
    pushed them off their straight route.
- Step 5, tests: `tests/test_site.py`, 12 tests. They cover the gates, routes, the wall constraint,
  the cell-list separation and body push against their all-pairs oracles, and the palace scene end
  to end at N = 200 (everyone leaves, never off walkable ground, deterministic). The full suite
  passes: 237 tests, 1 expected failure.
- 8,000-seat check, nearest plan:
  - everyone out in 27.1 min (T50 11.9 min, T95 24.9 min);
  - south gate: 6,800 people;
  - the slowest walkers' median exit is 15.4 min, against 11.6 min for everyone else.
- **v1 gate set, superseded by the gate audit:** 48,000 seats, everyone released at once (1 seed each; logs in `bench/logs/palace/`; figure
  `python -m figures.palace`, slide copy `docs/site/palace_egress_48000.png`):

  | Gate plan | Half out | 95% out | All out | Crowd time at Fruin F | Slowest walkers, median |
  |---|---|---|---|---|---|
  | Nearest gate (north 19,200, south 28,800; 3 gates unused) | 39.8 min | 95.7 min | 105.6 min | 7.36M cell-s | 48.8 min (others 39.1) |
  | Shared by gate width (all 5 public gates) | 26.3 min | 53.3 min | 85.4 min | 4.52M cell-s | 37.5 min (others 25.5) |
  | Shared by the gate rates measured in the run above (north 17,600, south 14,000, east 6,000 + 6,400, west 4,000) | 26.1 min | 49.5 min | **54.3 min** | 4.40M cell-s | 33.9 min (others 25.5) |

  - Using every public gate halves the time to get 95% out (95.7 to 53.3 min).
  - Sharing by width is not balanced in time. The west route squeezes through narrow passages
    between the palace's west wing and the compound wall, so its 4 m gate passed only 84 persons/min.
    That is 21 per metre, where the other gates sustained 41-47. The west gate's 8,000 people set
    the 85-minute tail.
  - The twin's own measurements give the third plan: north 17,600, south 14,000, east 6,000 + 6,400,
    west 4,000. It estimated about 50 min per gate. The run gave 45-54 min per gate and everyone out
    in 54.3 min, against 85.4 when sharing by width and 105.6 at the nearest gate. Measure, re-plan,
    check: that loop is what the twin is for.
  - What the simple formula cannot show: total width says about 40 min at the measured
    rate. The simulation shows where it fails, at an overloaded nearest gate and a throttled route.
  - These are comparisons under assumed gate widths and layout. Whether the west passages are
    really that narrow should be checked against the site before anyone acts on it.
- **Step 7, Unity port, done (2026-10-05).**
  - Data: `tools/export_site_unity.py` writes the grids, gates, seat-block candidates and footprints
    to `unity/Parity3D/Assets/StreamingAssets/palace/`.
  - Routing: `Sim/PalaceSite.cs` recomputes the gate fields and routes in C#. The smoke test checks
    them against Python's: worst field difference 5e-7 m, and all 120 routes are identical. All five
    fields take 0.9 s.
  - Scene: `Sim/PalaceScene.cs` holds seat blocks, release, row ends, block-side routes, the three gate
    plans (default: measured rates), the stall fallback, exits, and a reset loop for the demo. On
    passing a waypoint the core is re-seated from that waypoint, so a surrogate never cuts a
    building corner.
  - Set: `Demo/StagePalace.cs` extrudes every OSM footprint and building part to its mapped height
    (towers to 40 m; unmapped buildings 7 m, assumed), with domes on the tall parts, the compound
    wall broken at each gate under an arch, lawns and paving.
  - Demo and smoke test: there is a "palace" button in the demo HUD. The smoke test passes. With 800
    people over 2 minutes, nobody stands off walkable ground and everyone stands up and moves toward
    their gate. PARITY's worst drift is 0.00 (cap 3.95) against the baseline's 356.
  - Screenshots: `docs/site/palace_unity_split.png` (baseline left, with drift rings; PARITY right)
    and `palace_unity_aerial.png`, rendered with `ParityShots -shots palace`. At 8,000 people PARITY
    keeps every agent's behaviour live, where the baseline turns 6,800 into surrogates; worst drift
    27.4 against 0.00.
  - The engine moves its crowd with ORCA plus separation, not the Python exit model. The exit-time
    comparisons above come from `bench/palace.py`. The engine scene is the live, explorable twin
    that runs PARITY's allocator.
  - The cap and drift rates are provisional: the concourse's measured ones, until step 4.
- **Step 8, scale check, done.** Run with `ParityFrameBench -scene palace -sizes 8000,16000,32000,48000
  -targets 16 -knobs 16,1 -policies masslod,parity -overlap`. Apple M4, editor with Release code,
  1080p, MSAA 4, shadows; log `bench/logs/palace/frame_bench_palace_scale.*`.

  | People | MassLOD p95 | PARITY p95 (target 16 ms) | Worst drift / cap, MassLOD vs PARITY | CPU frame | Occlusion pass |
  |---|---|---|---|---|---|
  | 8,000 | 12.5 ms | **15.8 ms** | 4.20 vs 0.96 | 11.9 ms | 1.4 ms |
  | 16,000 | 21.4 ms | 22.6 ms | 4.20 vs 0.99 | 20.7 ms | 5.3 ms |
  | 32,000 | 39.2 ms | 41.1 ms | 4.48 vs 0.99 | 38.1 ms | 13.0 ms |
  | 48,000 | 57.2 ms | 59.1 ms | 4.40 vs 0.99 | 54.7 ms | 25.7 ms |

  - At 8,000 people PARITY meets the frame target with every agent within its drift cap, and
    with twice MassLOD's perceived quality (0.062 against 0.031).
  - Beyond about 10,000 people neither policy reaches 16 ms. The frame is CPU-bound by per-agent
    costs no allocator can cut: the occlusion-coverage pass, the core and draw preparation. The
    GPU waits only 0.1-0.15 ms per frame.
  - PARITY still holds every agent at or under its cap at every size (0.99 at most); MassLOD
    runs 4.2-4.5 times over. The drift guarantee scales; the frame rate does not, yet.
  - That points at the GPU work in the plan: the coverage pass and the core step as compute
    shaders, each with its C# version as the oracle (CLAUDE.md). Until then, the 48,000-seat
    study runs in `bench/palace.py`, and the live twin shows a sector (8,000 people) at full rate.
- **Demo video (2026-10-05):** `figures/out/palace_demo.mp4` (1080p, 2:11, 126 MB) and
  `palace_demo_720p.mp4` (19 MB; neither is in git).
  - Made by `ParityVideo` (Unity editor script, run on a clone of the project) and
    `python -m figures.palace_video`.
  - Contents: a title card; 45 s in real time of baseline | PARITY as 8,000 seated people stand up
    (baseline worst drift reaches 18 times its cap, PARITY's stays at 0.00); a 15x fast-forward from
    above as they walk to the gates; an end card with the 48,000-seat plan comparison.
  - **Known issue:** in the engine's ORCA crowd, 315 of the 1,600 people sent to the two 3 m east
    openings never get out, and 5 elsewhere. 7,680 of 8,000 are out by 13.75 min, then nobody
    leaves. The Python model clears the same gates, so this is a narrow-gate jam in the engine's
    motion, to fix with the gate work.
- **Known gaps against the real site (2026-10-05, from the team's review):**
  - The plan layout is real: footprints, spacing, walls, lawns, paths and gate positions come from
    OSM. The look is not: boxy extrusions, generic windows, sphere domes, no Indo-Saracenic detail.
  - Nothing outside the walls is modelled: roads, footpaths, parking.
  - Temporary Dasara barricades, screening lanes and seating enclosures are absent.
  - Gate roles are wrong for Jamboo Savari day. Police are posted at the Varaha (south), Brahmapuri
    and Karikal Thotti (both west) and Jayamarthanda (east) gates [Udayavani]. Entry is by pass,
    through door-frame metal detectors after frisking [Deccan Herald]. The model treats Jayamarthanda
    as closed, uses the north gate (the procession's exit) for spectators, and has only one west gate.
  - Gate widths are not published anywhere found so far.
- **Gate work (2026-10-05), step by step.**
  - **1. Audit:** section 5b.
  - **2. Gates in the model:** names, roles and estimated widths now come from the audit.
    - The spectator gates are Jayamarthanda (east), Varaha (south) and both west gates.
    - The north gates carry the procession and are closed to spectators unless a scenario opens them.
    - The two east forecourt side openings are in no report, so they are closed by default.
    - Earlier results used a corrected-away gate set: north open, Jayamarthanda shut, one west gate.
      They are kept above, marked v1.
  - **3. Arrival through screening** (`python -m bench.palace --mode entry`):
    - Pass holders arrive at the rim of their gate's approach zone. Arrivals are triangular over
      120 min with the mode 45 min before the start (ASSUMED).
    - They queue, and are screened on stepping into the gate passage: one lane per 1.5 m of
      width, 600 per hour per lane (section 5b). The person furthest forward goes next.
    - They then walk to their seat. The seated lean back to let late arrivals along their row.
    - When the rim is crowded (over 3 persons/m²), arrivals wait in the street beyond the model,
      and that spill is reported.
    - A screened walker the crowd pins against a wall for 5 s is put back on its route line, as
      invariant 4 binds a position to the core's route. The count is reported.
  - **4. Outside the gates:** each open gate has an approach zone, the open ground within 40 m outside
    it (outside buildings excluded). Leaving means reaching that zone's rim, not just passing the
    arch.
  - **5. Engine:** the export carries the audited gates and zones. Closed gates are drawn with their
    grille shut. The engine plan defaults to "balanced" until rates are measured for the new gate set.
- **Results with the audited gates (v2, 2026-10-05).** 48,000 people, one run each; logs
  `bench/logs/palace/v2_*`. Spectator gates: Jayamarthanda (10 m), Varaha (4.2 m), Brahmapuri? (4 m),
  Karikal Thotti? (3 m), 21.2 m of clear width in all. Leaving means reaching the rim of the 40 m
  approach zone.

  **Exit after the procession:**

  | Gate plan | Half out | 95% out | All out | Note |
  |---|---|---|---|---|
  | Nearest gate | 38.9 min | 148.7 min | 165.0 min | Varaha takes 29,600 people through 4.2 m |
  | Shared by gate width | 32.2 min | 74.2 min | 111.9 min | The two west gates set the tail; 23 people (0.05%) wedged at the 3 m gate |
  | Shared by measured gate rates | *running* | | | |
  | Width-shared, north gates also opened to spectators | *running* | | | |

  The audited gate set is much tighter than v1, which wrongly used the north gates for
  spectators. Even the better plan takes nearly two hours to clear.

  **Arrival before the procession** (entry is by pass through door-frame metal detectors with
  frisking). A fluid model of the stated assumptions (`python -m bench.palace --mode capacity`,
  DER, no crowd simulation) gives the headline:

  | Screening lanes x rate | Gates open before the start | Seated when the procession starts | Most waiting at once |
  |---|---|---|---|
  | 15 lanes x 600/h (1 lane per 1.5 m of gate) | 2 h | 33% | 31,700 |
  | 15 lanes x 600/h | 4 h | 61% | 18,700 |
  | 28 lanes x 600/h (1 per 0.75 m) | 3 h | 80% | 10,800 |
  | 28 lanes x 600/h | 4 h | 96% | 4,300 |
  | 15 lanes x 1,200/h | 4 h | 99% | 3,000 |

  - The crowd simulation of the first row agrees: 29.9% seated when the procession starts. That
    run used nearest-gate pass assignment and no cap on the approach zones, so its densities are
    not physical. Reruns with the zone cap and capacity-shared passes are under way.
  - **Planning reading:** at Jamboo Savari scale, the number of screening lanes and how early the
    gates open decide whether people are seated in time and how many wait in the street. Gate width
    matters much less on the way in. On the way out, without screening, width and the gate plan
    decide.
- **Engine check on the audited gates:** 8,000 people, width-shared plan. 7,992 out by 18 min;
  8 stalled at the 3 m west gate (it was 320 with v1's gates). A second-stage nudge (one cell
  along the field) is now in both models; a rerun is under way.
- Step 4 (calibration through the tiered pipeline) is moved after the checkpoint. It needs the
  invariant core generalised to routes with more than three waypoints and one capacity counter
  per gate, which changes code under every existing scene.
