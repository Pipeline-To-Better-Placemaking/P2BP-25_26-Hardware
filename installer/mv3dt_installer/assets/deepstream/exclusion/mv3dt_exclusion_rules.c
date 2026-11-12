/*
 * mv3dt_exclusion_rules.c - config parsing and the inside rule for the
 * detection exclusion filter. GLib only.
 *
 * Config: a GKeyFile with one group per source index, each with a `rects`
 * list of "left,top,width,height" entries in streammux pixels, separated
 * by ';'. Numbers are non-negative decimal integers; width and height are
 * greater than zero; blanks around numbers and entries and empty entries
 * are allowed. Frame bounds are not checked here.
 * Example:
 *
 *   [source1]
 *   rects=530,175,245,225
 *
 * A group whose rects fail to parse is ignored as a whole. Group names
 * other than source<N> with N below MV3DT_EXCLUSION_MAX_SOURCES are ignored.
 */

#include <string.h>

#include "mv3dt_exclusion_rules.h"

/* Parse one non-negative decimal integer: digits only, no sign and no
 * fraction, surrounding blanks allowed (section 4.2). */
static gboolean
parse_number (gchar * text, gdouble * out)
{
  gchar *s = g_strstrip (text);
  guint64 n = 0;
  if (!g_ascii_isdigit (*s))
    return FALSE;
  for (const gchar * c = s; *c; c++)
    if (!g_ascii_isdigit (*c))
      return FALSE;
  if (!g_ascii_string_to_unsigned (s, 10, 0, G_MAXUINT32, &n, NULL))
    return FALSE;
  *out = (gdouble) n;
  return TRUE;
}

/* Parse a rects value into out. FALSE on any bad entry; out is then left
 * partly filled and the caller discards it. Empty entries are skipped. */
static gboolean
parse_rects (const gchar * value, GArray * out)
{
  gchar **entries = g_strsplit (value, ";", -1);
  gboolean ok = TRUE;
  for (gchar ** e = entries; ok && *e; e++) {
    gchar *entry = g_strstrip (*e);
    if (!*entry)
      continue;
    gchar **parts = g_strsplit (entry, ",", -1);
    gdouble v[4] = { 0, 0, 0, 0 };
    if (g_strv_length (parts) != 4) {
      ok = FALSE;
    } else {
      for (int i = 0; ok && i < 4; i++)
        ok = parse_number (parts[i], &v[i]);
    }
    g_strfreev (parts);
    if (ok && (v[2] <= 0 || v[3] <= 0))
      ok = FALSE;
    if (ok) {
      Mv3dtExclusionRect r = { (gfloat) v[0], (gfloat) v[1],
        (gfloat) (v[0] + v[2]), (gfloat) (v[1] + v[3])
      };
      g_array_append_val (out, r);
    }
  }
  g_strfreev (entries);
  return ok;
}

/* The N of a group named source<N>, or -1 when the name does not match or
 * N is out of range. */
static gint
source_index (const gchar * group)
{
  const gchar *digits;
  guint64 n = 0;
  if (!g_str_has_prefix (group, "source"))
    return -1;
  digits = group + strlen ("source");
  if (!*digits || !g_ascii_isdigit (*digits))
    return -1;
  if (!g_ascii_string_to_unsigned (digits, 10, 0,
          MV3DT_EXCLUSION_MAX_SOURCES - 1, &n, NULL))
    return -1;
  return (gint) n;
}

static Mv3dtExclusionRules *
rules_from_key_file (GKeyFile * kf)
{
  Mv3dtExclusionRules *rules = g_new0 (Mv3dtExclusionRules, 1);
  gsize n_groups = 0;
  gchar **groups = g_key_file_get_groups (kf, &n_groups);
  for (gsize g = 0; g < n_groups; g++) {
    gint index = source_index (groups[g]);
    if (index < 0) {
      g_printerr ("mv3dt-exclusion: ignoring group [%s]\n", groups[g]);
      continue;
    }
    gchar *value = g_key_file_get_string (kf, groups[g], "rects", NULL);
    if (!value) {
      g_printerr ("mv3dt-exclusion: no rects in [%s]; group ignored\n",
          groups[g]);
      continue;
    }
    GArray *rects = g_array_new (FALSE, FALSE, sizeof (Mv3dtExclusionRect));
    if (!parse_rects (value, rects)) {
      g_printerr ("mv3dt-exclusion: bad rects in [%s]: %s; group ignored\n",
          groups[g], value);
      g_array_free (rects, TRUE);
    } else if (rects->len == 0) {
      g_array_free (rects, TRUE);
    } else {
      if (rules->rects[index])
        g_array_free (rules->rects[index], TRUE);
      rules->rects[index] = rects;
    }
    g_free (value);
  }
  g_strfreev (groups);
  for (guint s = 0; s < MV3DT_EXCLUSION_MAX_SOURCES; s++)
    if (rules->rects[s])
      rules->total += rules->rects[s]->len;
  return rules;
}

Mv3dtExclusionRules *
mv3dt_exclusion_rules_load_file (const gchar * path, GError ** error)
{
  GKeyFile *kf = g_key_file_new ();
  Mv3dtExclusionRules *rules = NULL;
  if (g_key_file_load_from_file (kf, path, G_KEY_FILE_NONE, error))
    rules = rules_from_key_file (kf);
  g_key_file_free (kf);
  return rules;
}

Mv3dtExclusionRules *
mv3dt_exclusion_rules_load_data (const gchar * data, gssize length,
    GError ** error)
{
  GKeyFile *kf = g_key_file_new ();
  Mv3dtExclusionRules *rules = NULL;
  gsize len = length < 0 ? strlen (data) : (gsize) length;
  if (g_key_file_load_from_data (kf, data, len, G_KEY_FILE_NONE, error))
    rules = rules_from_key_file (kf);
  g_key_file_free (kf);
  return rules;
}

void
mv3dt_exclusion_rules_free (Mv3dtExclusionRules * rules)
{
  if (!rules)
    return;
  for (guint s = 0; s < MV3DT_EXCLUSION_MAX_SOURCES; s++)
    if (rules->rects[s])
      g_array_free (rules->rects[s], TRUE);
  g_free (rules);
}

const GArray *
mv3dt_exclusion_rules_for_source (const Mv3dtExclusionRules * rules,
    guint source)
{
  if (!rules || source >= MV3DT_EXCLUSION_MAX_SOURCES)
    return NULL;
  return rules->rects[source];
}

gboolean
mv3dt_exclusion_rect_contains (const Mv3dtExclusionRect * r, gfloat left,
    gfloat top, gfloat width, gfloat height)
{
  return left >= r->left && top >= r->top
      && left + width <= r->right && top + height <= r->bottom;
}

gboolean
mv3dt_exclusion_rect_overlaps (const Mv3dtExclusionRect * r, gfloat left,
    gfloat top, gfloat width, gfloat height)
{
  return left < r->right && left + width > r->left
      && top < r->bottom && top + height > r->top;
}

gboolean
mv3dt_exclusion_rects_inside_any (const GArray * rects, gfloat left,
    gfloat top, gfloat width, gfloat height)
{
  if (!rects)
    return FALSE;
  for (guint i = 0; i < rects->len; i++)
    if (mv3dt_exclusion_rect_contains (&g_array_index (rects,
                Mv3dtExclusionRect, i), left, top, width, height))
      return TRUE;
  return FALSE;
}

gboolean
mv3dt_exclusion_rects_overlap_any (const GArray * rects, gfloat left,
    gfloat top, gfloat width, gfloat height)
{
  if (!rects)
    return FALSE;
  for (guint i = 0; i < rects->len; i++)
    if (mv3dt_exclusion_rect_overlaps (&g_array_index (rects,
                Mv3dtExclusionRect, i), left, top, width, height))
      return TRUE;
  return FALSE;
}
