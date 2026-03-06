import json
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from pathlib import Path
from collections import defaultdict

# FUSION PROBLEMS AND BREAKDOWN 
#
# The similarity matrix for this footage shows scores spanning 0.47–0.74 with
# NO natural gap between same-person and different-person pairs. The mean
# cross-person similarity is approx. 0.62, which means a threshold-based approach
# using appearance alone will always either over-merge or under-merge.
#
# Solution: use STRUCTURAL constraints (time gap + position) as the PRIMARY
# gate. Appearance is used only as a relative tie-breaker — we ask "is this
# the BEST candidate" rather than "does this pass a fixed threshold."
#
# CONFIG
# max_gap_s          : Max time gap (seconds) to consider two tracks the same
#                      person. Currently a tight constraint: if the gap is large, different
#                      people may have entered/exited in the meantime.

# max_position_px    : Max pixel distance between the end of one track and the
#                      start of the next. Prevents teleportation merges.

# min_appearance_sim : Minimum appearance similarity floor. A pair that passes
#                      time+position gates but scores below this is still
#                      rejected. Set LOW (e.g. 0.55) because person scores avg around 0.62,
#                      it blocks clear misses and allows fusion matches.

# max_overlap_frames : How many frames two tracks can share and still be merged
#                      as a "short duplicate". If both tracks are long and
#                      overlapping then they are highly likely two different people.

# min_track_length   : Tracks shorter than this are noise/quick ID switches to ignore. 

class PostProcessIDFusion:
    def __init__(self,
                 max_gap_s=6.0,
                 max_position_px=150,
                 min_appearance_sim=0.55,
                 max_overlap_s=2.0,
                 min_track_length=10):

        self.max_gap_s = max_gap_s
        self.max_position_px = max_position_px
        self.min_appearance_sim = min_appearance_sim
        self.max_overlap_s = max_overlap_s
        self.min_track_length = min_track_length

    # Loading in JSON file tracking data and outputting fused data 

    def load_tracking_data(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} tracks from {json_file}")
        return data

    def save_fused_data(self, fused_data, output_file):
        output_data = {}
        for track_key, track_data in fused_data.items():
            output_data[track_key] = {
                'num_detections': track_data['num_detections'],
                'duration': track_data['duration'],
                'num_features': track_data['num_features'],
                'features': track_data['features'],
                'original_track_ids': track_data.get('original_track_ids',
                                                      [int(track_key.split('_')[1])])
            }
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Fused tracking data saved to: {output_file}")

    # Similarity and distance helper functions 

    def _cosine_sim(self, v1, v2):
        return float(1 - cosine(v1, v2))

    def _bbox_centre(self, bbox):
        return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

    def _position_dist(self, bbox1, bbox2):
        if bbox1 is None or bbox2 is None:
            return float('inf')
        return float(euclidean(self._bbox_centre(bbox1), self._bbox_centre(bbox2)))

    def _endpoint_sim(self, td1, td2):
        """
        Similarity between the tail of track1 and the head of track2.
        Averages the last 3 features of td1 vs the first 3 of td2 for
        robustness against single bad crops.
        """
        tail = td1['features'][-min(3, len(td1['features'])):]
        head = td2['features'][:min(3, len(td2['features']))]
        sims = [
            self._cosine_sim(
                np.array(a['feature_vector']),
                np.array(b['feature_vector'])
            )
            for a in tail for b in head
        ]
        return float(np.median(sims)) if sims else 0.0

    def _full_track_sim(self, feats1, feats2, n=6):
        """Median pairwise similarity sampling n frames from each track.
        Accepts feature lists directly (not full track dicts)."""
        def sample(feats, n):
            if len(feats) <= n:
                return feats
            idx = np.linspace(0, len(feats) - 1, n, dtype=int)
            return [feats[i] for i in idx]

        s1 = sample(feats1, n)
        s2 = sample(feats2, n)
        sims = [
            self._cosine_sim(np.array(a['feature_vector']), np.array(b['feature_vector']))
            for a in s1 for b in s2
        ]
        return float(np.median(sims)) if sims else 0.0

    # Track data functions for time range and overlap

    def _time_range(self, td):
        ts = [f['timestamp'] for f in td['features']]
        return min(ts), max(ts)

    def _time_overlap_s(self, td1, td2):
        """
        Overlap in seconds between the detection time ranges of two tracks.
        More reliable than frame-set intersection when features are sparse (1/s).
        """
        s1, e1 = self._time_range(td1)
        s2, e2 = self._time_range(td2)
        return max(0.0, min(e1, e2) - max(s1, s2))

    # Union find structure for merging tracks while respecting exclusions

    def _build_union_find(self, n):
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                if ra < rb:
                    parent[rb] = ra
                else:
                    parent[ra] = rb

        return find, union

    # Exclusion set construction: identify pairs of tracks that should NOT be merged/are not the same person

    def _build_exclusion_set(self, valid):
        """
        Two tracks are mutually exclusive (different people) if they overlap
        substantially in time AND look sufficiently different from each other.

        Two cases:
          High overlap + LOW sim  -> different people co-present   -> EXCLUDE
          High overlap + HIGH sim -> same person double-detected   -> DUPLICATE (allow merge)

        duplicate_sim_threshold is set high (0.65) to avoid accidentally merging
        genuinely different people who happen to look similar.
        """
        keys = list(valid.keys())
        excluded = set()
        duplicates = set()
        # We can tell a genuine YOLO duplicate detection from two factors: 
        #   1. Short overlap where YOLO briefly splits one person into two boxes (typically <3s)
        #   2. High similarity where both boxes are literally the same person
        
        # A long overlap with high sim is more likely two similar-looking people
        # walking together, so we treat those as exclusions too.
        duplicate_sim_threshold = 0.67
        max_duplicate_overlap_s = 3.0   # overlaps longer than this are not YOLO duplicates

        for i, k1 in enumerate(keys):
            for k2 in keys[i + 1:]:
                overlap_s = self._time_overlap_s(valid[k1], valid[k2])
                if overlap_s <= self.max_overlap_s:
                    continue

                sim = self._full_track_sim(valid[k1]['features'], valid[k2]['features'])
                pair = (min(k1, k2), max(k1, k2))

                is_short_overlap = overlap_s <= max_duplicate_overlap_s
                is_high_sim = sim >= duplicate_sim_threshold

                if is_short_overlap and is_high_sim:
                    duplicates.add(pair)
                    print(f"    Duplicate: {k1} <-> {k2}  overlap={overlap_s:.1f}s  sim={sim:.3f} -> allow merge")
                else:
                    excluded.add(pair)
                    reason = []
                    if not is_short_overlap: reason.append(f"overlap {overlap_s:.1f}s > {max_duplicate_overlap_s}s")
                    if not is_high_sim: reason.append(f"sim {sim:.3f} < {duplicate_sim_threshold}")
                    print(f"    Exclude:   {k1} <-> {k2}  overlap={overlap_s:.1f}s  sim={sim:.3f} -> block ({', '.join(reason)})")

        print(f"  Exclusion pairs (different people): {len(excluded)}")
        print(f"  Duplicate pairs (same person, YOLO double-detection): {len(duplicates)}")
        return excluded, duplicates

    # Main fusion pipeline: find candidates, apply fusions, absorb isolated tracks, generate report

    def find_fusion_candidates(self, tracking_data):
        """
        For each track that ends, find the BEST continuation among all tracks
        that start within the time/position window.

        Key difference from threshold-based approach: instead of asking
        "does this pair score above X?", we ask "among all valid continuations
        for this track, which scores highest?" — then accept it if it clears
        the soft appearance floor (min_appearance_sim).

        This prevents the case where a mediocre match at 0.62 wins because
        the true match is also at 0.64 — both would pass a fixed threshold,
        but only the best one gets selected here.
        """
        valid = {k: v for k, v in tracking_data.items()
                 if v['num_detections'] >= self.min_track_length and v['features']}

        dropped = len(tracking_data) - len(valid)
        print(f"\nAnalysing {len(valid)} valid tracks (dropped {dropped} noise tracks)...")

        excluded, duplicates = self._build_exclusion_set(valid)

        # Sort by start time
        track_keys = sorted(valid.keys(),
                            key=lambda k: valid[k]['features'][0]['timestamp'])

        candidates = []
        seen_pairs = set()

        # Add duplicate pairs as merge candidates (same-person YOLO double-detections)
        for (k1, k2) in duplicates:
            sim = self._full_track_sim(valid[k1]['features'], valid[k2]['features'])
            candidates.append({'key1': k1, 'key2': k2, 'similarity': sim,
                                'time_gap': 0, 'method': 'duplicate_detection'})
            seen_pairs.add((min(k1, k2), max(k1, k2)))
            print(f"  Duplicate merge queued: {k1} + {k2}  sim={sim:.3f}")

        def try_add(k1, k2, sim, gap, method):
            pair = (min(k1, k2), max(k1, k2))
            if pair not in seen_pairs and pair not in excluded:
                seen_pairs.add(pair)
                candidates.append({
                    'key1': k1, 'key2': k2,
                    'similarity': sim,
                    'time_gap': gap,
                    'method': method
                })
                return True
            return False

        print(f"\nPass 1: Best-continuation matching (time + position + appearance)...")
        for i, k1 in enumerate(track_keys):
            td1 = valid[k1]
            _, t1_end = self._time_range(td1)
            last_feat = td1['features'][-1]

            continuation_pool = []

            for k2 in track_keys[i + 1:]:
                td2 = valid[k2]
                t2_start, _ = self._time_range(td2)
                gap = t2_start - t1_end

                if gap < -0.5:
                    continue
                if gap > self.max_gap_s:
                    break

                pair = (min(k1, k2), max(k1, k2))
                if pair in excluded:
                    continue

                first_feat = td2['features'][0]
                if last_feat.get('bbox') and first_feat.get('bbox'):
                    pos_dist = self._position_dist(last_feat['bbox'], first_feat['bbox'])
                    if pos_dist > self.max_position_px:
                        print(f"    Position blocked: {k1}->{k2}  dist={pos_dist:.0f}px > {self.max_position_px}px")
                        continue
                else:
                    pos_dist = None

                sim = self._endpoint_sim(td1, td2)
                continuation_pool.append((k2, sim, gap, pos_dist))

            if not continuation_pool:
                continue

            best_k2, best_sim, best_gap, best_pos = max(continuation_pool, key=lambda x: x[1])

            if best_sim >= self.min_appearance_sim:
                added = try_add(k1, best_k2, best_sim, best_gap, 'best_continuation')
                if added:
                    pos_str = f"  pos={best_pos:.0f}px" if best_pos is not None else ""
                    print(f"  Continuation: {k1} -> {best_k2}  sim={best_sim:.3f}  gap={best_gap:.2f}s{pos_str}")
            else:
                print(f"  Rejected best for {k1}: {best_k2}  sim={best_sim:.3f} < {self.min_appearance_sim}")

        print(f"\nPass 2: Short concurrent duplicates (YOLO double-detection)...")
        for i, k1 in enumerate(track_keys):
            for k2 in track_keys[i + 1:]:
                pair = (min(k1, k2), max(k1, k2))
                if pair in seen_pairs or pair in excluded:
                    continue

                td1, td2 = valid[k1], valid[k2]
                overlap_s = self._time_overlap_s(td1, td2)
                if overlap_s == 0:
                    continue

                shorter = min(td1['num_detections'], td2['num_detections'])
                longer = max(td1['num_detections'], td2['num_detections'])
                if shorter > 20 or longer > 60:
                    continue

                sim = self._full_track_sim(td1['features'], td2['features'])
                if sim >= 0.68:
                    try_add(k1, k2, sim, 0, 'concurrent_duplicate')
                    print(f"  Concurrent dup: {k1} ↔ {k2}  sim={sim:.3f}  overlap={overlap_s:.1f}s")

        return candidates, valid, excluded, duplicates

    def apply_fusions(self, tracking_data, candidates, valid_tracks, excluded, duplicates):
        """
        Apply fusions with union find and exclusion rules
        if a proposed union would join two mutually-exclusive tracks
        (i.e. they were simultaneously visible), block it.
        """
        all_keys = list(tracking_data.keys())
        key_to_idx = {k: i for i, k in enumerate(all_keys)}
        find, union = self._build_union_find(len(all_keys))

        candidates.sort(key=lambda c: c['similarity'], reverse=True)

        print(f"\nApplying {len(candidates)} fusion candidates...")
        blocked = 0
        for c in candidates:
            k1, k2 = c['key1'], c['key2']
            if k1 not in key_to_idx or k2 not in key_to_idx:
                continue

            root1 = find(key_to_idx[k1])
            root2 = find(key_to_idx[k2])
            if root1 == root2:
                continue

            group1_keys = [k for k in all_keys if find(key_to_idx[k]) == root1]
            group2_keys = [k for k in all_keys if find(key_to_idx[k]) == root2]

            conflict = any(
                (min(gk1, gk2), max(gk1, gk2)) in excluded
                for gk1 in group1_keys
                for gk2 in group2_keys
            )

            # Duplicate detections bypass exclusion: identified as same person overlaps
            is_duplicate = c['method'] == 'duplicate_detection'
            if conflict and not is_duplicate:
                blocked += 1
                print(f"  BLOCKED (exclusion conflict): {k1} + {k2}  [{c['method']}  sim={c['similarity']:.3f}]")
            else:
                union(key_to_idx[k1], key_to_idx[k2])
                print(f"  Merge {k1} + {k2}  [{c['method']}  sim={c['similarity']:.3f}]")

        print(f"\n  Blocked by exclusion: {blocked}")

        groups = defaultdict(list)
        for key in all_keys:
            groups[find(key_to_idx[key])].append(key)

        print(f"  {len(tracking_data)} original tracks -> {len(groups)} fused groups")

        fused_data = {}
        for root, keys in groups.items():
            master_key = min(keys, key=lambda k: int(k.split('_')[1]))
            merged_features = []
            total_detections = 0
            original_ids = []

            for k in sorted(keys, key=lambda x: int(x.split('_')[1])):
                if k in tracking_data:
                    merged_features.extend(tracking_data[k]['features'])
                    total_detections += tracking_data[k]['num_detections']
                    original_ids.append(int(k.split('_')[1]))

            merged_features.sort(key=lambda x: x['timestamp'])
            duration = (merged_features[-1]['timestamp'] - merged_features[0]['timestamp']
                        if len(merged_features) > 1 else 0)

            fused_data[master_key] = {
                'num_detections': total_detections,
                'duration': duration,
                'num_features': len(merged_features),
                'features': merged_features,
                'original_track_ids': original_ids
            }

        return fused_data

    def absorb_orphans(self, fused_data, original_data, excluded):
        """
        After main fusion, assign isolated single-track fragments to their
        correct person group by process of elimination and appearance

          1. Classify each fused group as "established" (multi-track or large)
             vs "orphan" (single original track, small)
          2. For each orphan, find all established groups it is NOT excluded from
          3. Among those valid groups, pick the highest appearance similarity
          4. If no established group is valid, try merging orphans with each other
             (same logic, only if not mutually excluded)
        """
        result = dict(fused_data)

        def get_orig_ids(td, key):
            return set(td.get('original_track_ids', [int(key.split('_')[1])]))

        def is_excluded(orig_ids_a, orig_ids_b):
            for oa in orig_ids_a:
                for ob in orig_ids_b:
                    ka = f'track_{oa}'
                    kb = f'track_{ob}'
                    pair = (min(ka, kb), max(ka, kb))
                    if pair in excluded:
                        return True
            return False

        def classify(result):
            orphans = {}
            established = {}
            for key, td in result.items():
                orig_ids = get_orig_ids(td, key)
                is_orphan = (len(orig_ids) == 1 and
                             td['num_detections'] < 500 and
                             len(td.get('features', [])) > 0)
                if is_orphan:
                    orphans[key] = td
                else:
                    established[key] = td
            return orphans, established

        def absorb_one_round(result):
            orphans, established = classify(result)
            if not orphans:
                return result, 0

            absorbed = 0
            # Sort largest orphan first, which then becomes anchor for smaller ones
            for o_key in sorted(orphans, key=lambda k: result[k]['num_detections'], reverse=True):
                if o_key not in result:
                    continue
                o_td = result[o_key]
                o_orig = get_orig_ids(o_td, o_key)
                o_feats = o_td.get('features', [])
                if not o_feats:
                    continue

                best_key = None
                best_sim = self.min_appearance_sim  # must beat floor

                # Re classify on each iteration so absorbed orphans become candidates
                _, current_established = classify(result)

                for g_key, g_td in current_established.items():
                    if g_key == o_key:
                        continue
                    g_orig = get_orig_ids(g_td, g_key)
                    g_feats = g_td.get('features', [])
                    if not g_feats:
                        continue
                    if is_excluded(o_orig, g_orig):
                        continue
                    sim = self._full_track_sim(o_feats, g_feats)
                    if sim > best_sim:
                        best_sim = sim
                        best_key = g_key

                if best_key:
                    g = result[best_key]
                    merged = sorted(g['features'] + o_feats, key=lambda x: x['timestamp'])
                    new_orig = sorted(get_orig_ids(g, best_key) | o_orig)
                    duration = (merged[-1]['timestamp'] - merged[0]['timestamp']) if len(merged) > 1 else 0
                    result[best_key] = {
                        'num_detections': g['num_detections'] + o_td['num_detections'],
                        'duration': duration,
                        'num_features': len(merged),
                        'features': merged,
                        'original_track_ids': new_orig
                    }
                    del result[o_key]
                    absorbed += 1
                    print(f"    Absorb {o_key} -> {best_key}  sim={best_sim:.3f}")

            return result, absorbed

        orphans_initial, established_initial = classify(result)
        print(f"  Orphan absorption: {len(orphans_initial)} orphan(s), "
              f"{len(established_initial)} established group(s)")

        # Run multiple rounds until no more absorptions happen
        for round_num in range(5):
            result, absorbed = absorb_one_round(result)
            if absorbed == 0:
                break

        # Report any remaining orphans
        orphans_remaining, _ = classify(result)
        for key in orphans_remaining:
            print(f"    Unabsorbed: {key} — excluded from all groups, keeping as separate ID")

        return result

    def generate_report(self, original_data, fused_data, output_file=None):
        lines = ["=" * 70, "ID FUSION REPORT", "=" * 70]
        lines.append(f"Original tracks : {len(original_data)}")
        lines.append(f"Fused tracks    : {len(fused_data)}")
        lines.append(f"Tracks merged   : {len(original_data) - len(fused_data)}")
        lines.append("")
        lines.append("-" * 70)

        for key in sorted(fused_data.keys(), key=lambda k: int(k.split('_')[1])):
            td = fused_data[key]
            orig_ids = td.get('original_track_ids', [int(key.split('_')[1])])
            lines.append(f"\nGlobal ID {key.split('_')[1]}:")
            lines.append(f"  Fused from : {orig_ids}" if len(orig_ids) > 1
                         else f"  Original ID: {orig_ids[0]}")
            lines.append(f"  Detections : {td['num_detections']}")
            lines.append(f"  Duration   : {td['duration']:.2f}s")
            lines.append(f"  Features   : {td['num_features']}")
            if td['features']:
                t0 = td['features'][0]['timestamp']
                t1 = td['features'][-1]['timestamp']
                lines.append(f"  Time range : {t0:.2f}s – {t1:.2f}s")

        report = "\n".join(lines)
        print("\n" + report)
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\nReport saved to: {output_file}")
        return report



def process_video(input_json,
                  output_json=None,
                  max_gap_s=6.0,
                  max_position_px=150,
                  min_appearance_sim=0.55,
                  max_overlap_s=2.0,
                  min_track_length=10):
    """
    All fusion steps/pipeline: load file -> find candidates -> fuse tracks -> save -> report.

    Tuning and fixing fusion: 
    
    Still over-merging (too few IDs):
      • Lower  max_gap_s           (e.g. 3.0)
      • Lower  max_position_px     (e.g. 100)
      • Raise  min_appearance_sim  (e.g. 0.60)
      • Lower  max_overlap_s       (e.g. 1.0)  ← tighter co-presence = more exclusions

    Under-merging (too many IDs):
      • Raise  max_gap_s           (e.g. 8.0)
      • Raise  max_position_px     (e.g. 200)
      • Lower  min_appearance_sim  (e.g. 0.50)
      • Raise  max_overlap_s       (e.g. 4.0)
    """
    
    fusion = PostProcessIDFusion(
        max_gap_s=max_gap_s,
        max_position_px=max_position_px,
        min_appearance_sim=min_appearance_sim,
        max_overlap_s=max_overlap_s,
        min_track_length=min_track_length
    )

    print(f"\n{'=' * 70}\nProcessing: {input_json}\n{'=' * 70}")

    original_data = fusion.load_tracking_data(input_json)
    candidates, valid_tracks, excluded, duplicates = fusion.find_fusion_candidates(original_data)

    if not candidates:
        print("\nNo fusion candidates found.")
        return original_data

    fused_data = fusion.apply_fusions(original_data, candidates, valid_tracks, excluded, duplicates)

    if output_json is None:
        p = Path(input_json)
        output_json = p.parent / f"{p.stem}_fused.json"

    print("\nOrphan absorption pass...")
    fused_data = fusion.absorb_orphans(fused_data, original_data, excluded)

    fusion.save_fused_data(fused_data, output_json)
    report_file = Path(output_json).parent / f"{Path(output_json).stem}_report.txt"
    fusion.generate_report(original_data, fused_data, report_file)

    return fused_data


if __name__ == "__main__":
    print("Post-Processing ID Fusion")
    print("=" * 70)

    input_file = 'tracking_outputs/4p-c0-trim_improved_features222_20260305_205548.json'

    fused = process_video(
        input_file,
        max_gap_s=6.0,           # max seconds between track end and next track start
        max_position_px=150,     # max pixel distance for position continuity check
        min_appearance_sim=0.55, # soft floor on appearance
        max_overlap_s=2.0,       # tracks overlapping more than this in time = different people
        min_track_length=10,
    )