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
- **48,000 seats, everyone released at once** (1 seed each; logs in `bench/logs/palace/`; figure
  `python -m figures.palace`):

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
- Step 4 (calibration through the tiered pipeline) is moved after the checkpoint. It needs the
  invariant core generalised to routes with more than three waypoints and one capacity counter
  per gate, which changes code under every existing scene.
