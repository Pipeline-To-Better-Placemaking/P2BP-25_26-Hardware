/*
 * mv3dt_exclusion_rules.h - parse the detection exclusion config into
 * per-source rectangle lists, and test boxes against them.
 *
 * GLib only: no DeepStream or GStreamer headers, so this compiles in tests.
 * The config format is DETECTION-EXCLUSION-ZONES section 4.2; the inside
 * rule is section 3.2.
 */
#ifndef MV3DT_EXCLUSION_RULES_H
#define MV3DT_EXCLUSION_RULES_H

#include <glib.h>

G_BEGIN_DECLS

/* Groups are [source<N>] with N below this bound. */
#define MV3DT_EXCLUSION_MAX_SOURCES 64

/* One exclusion rectangle in streammux pixels, stored as edges. */
typedef struct
{
  gfloat left, top, right, bottom;
} Mv3dtExclusionRect;

typedef struct
{
  /* Mv3dtExclusionRect per source index; NULL when the source has none. */
  GArray *rects[MV3DT_EXCLUSION_MAX_SOURCES];
  /* Number of rectangles over all sources. */
  guint total;
} Mv3dtExclusionRules;

/* Parse a config file. Returns NULL and sets error when the file cannot be
 * read or is not a key file. Bad or unknown groups are logged to stderr and
 * skipped, so the result may hold no rectangles (total == 0). */
Mv3dtExclusionRules *mv3dt_exclusion_rules_load_file (const gchar * path,
    GError ** error);

/* As mv3dt_exclusion_rules_load_file, from text in memory. length may be
 * -1 for a NUL-terminated string. */
Mv3dtExclusionRules *mv3dt_exclusion_rules_load_data (const gchar * data,
    gssize length, GError ** error);

void mv3dt_exclusion_rules_free (Mv3dtExclusionRules * rules);

/* The rectangles of one source, or NULL when it has none. */
const GArray *mv3dt_exclusion_rules_for_source (const Mv3dtExclusionRules *
    rules, guint source);

/* TRUE when the box lies entirely inside r, edges inclusive. */
gboolean mv3dt_exclusion_rect_contains (const Mv3dtExclusionRect * r,
    gfloat left, gfloat top, gfloat width, gfloat height);

/* TRUE when the box and r share any area. Used only for debug logging. */
gboolean mv3dt_exclusion_rect_overlaps (const Mv3dtExclusionRect * r,
    gfloat left, gfloat top, gfloat width, gfloat height);

/* TRUE when the box lies entirely inside any rectangle of the list. */
gboolean mv3dt_exclusion_rects_inside_any (const GArray * rects,
    gfloat left, gfloat top, gfloat width, gfloat height);

/* TRUE when the box overlaps any rectangle of the list. */
gboolean mv3dt_exclusion_rects_overlap_any (const GArray * rects,
    gfloat left, gfloat top, gfloat width, gfloat height);

G_END_DECLS

#endif
