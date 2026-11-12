/*
 * mv3dt_exclusion.c - drop detector boxes that lie entirely inside a
 * per-source exclusion rectangle, before they reach nvtracker.
 *
 * The config is named by $MV3DT_EXCLUSION_CONFIG and parsed by
 * mv3dt_exclusion_rules.c. An unset variable, an unreadable file or a file
 * without valid rectangles disables the filter with one log line; the
 * pipeline always runs.
 *
 * $MV3DT_EXCLUSION_DEBUG=1 logs each dropped box, and each kept box that
 * overlaps a rectangle, up to MV3DT_DEBUG_MAX_LINES lines.
 */

#include <glib.h>
#include <gst/gst.h>

#include "gstnvdsmeta.h"
#include "mv3dt_exclusion.h"
#include "mv3dt_exclusion_rules.h"

#define MV3DT_LOG_INTERVAL_US (60 * G_USEC_PER_SEC)
#define MV3DT_DEBUG_MAX_LINES 200

typedef struct
{
  Mv3dtExclusionRules *rules;
  guint64 dropped[MV3DT_EXCLUSION_MAX_SOURCES];
  gint64 last_log_us;
  gboolean debug;
  guint debug_logged;
} ExclusionFilter;

static void
filter_free (gpointer data)
{
  ExclusionFilter *f = data;
  mv3dt_exclusion_rules_free (f->rules);
  g_free (f);
}

static ExclusionFilter *
load_filter (const gchar * path)
{
  GError *err = NULL;
  Mv3dtExclusionRules *rules = mv3dt_exclusion_rules_load_file (path, &err);
  if (!rules) {
    g_printerr ("mv3dt-exclusion: cannot read %s: %s; filter disabled\n",
        path, err ? err->message : "unknown error");
    g_clear_error (&err);
    return NULL;
  }
  if (!rules->total) {
    g_print ("mv3dt-exclusion: %s has no rectangles; filter disabled\n",
        path);
    mv3dt_exclusion_rules_free (rules);
    return NULL;
  }
  for (guint s = 0; s < MV3DT_EXCLUSION_MAX_SOURCES; s++) {
    const GArray *rects = rules->rects[s];
    if (!rects)
      continue;
    for (guint i = 0; i < rects->len; i++) {
      const Mv3dtExclusionRect *r =
          &g_array_index (rects, Mv3dtExclusionRect, i);
      g_print ("mv3dt-exclusion: source %u excludes boxes inside "
          "left=%.0f top=%.0f right=%.0f bottom=%.0f\n", s, r->left, r->top,
          r->right, r->bottom);
    }
  }
  ExclusionFilter *f = g_new0 (ExclusionFilter, 1);
  f->rules = rules;
  f->last_log_us = g_get_monotonic_time ();
  f->debug = g_strcmp0 (g_getenv ("MV3DT_EXCLUSION_DEBUG"), "1") == 0;
  return f;
}

static void
debug_box (ExclusionFilter * f, const gchar * verdict, guint source,
    gint frame_num, const NvDsObjectMeta * obj, const gchar * suffix)
{
  if (!f->debug || f->debug_logged >= MV3DT_DEBUG_MAX_LINES)
    return;
  g_print ("mv3dt-exclusion: %s source %u frame %d box "
      "left=%.0f top=%.0f width=%.0f height=%.0f conf=%.2f%s\n", verdict,
      source, frame_num, obj->rect_params.left, obj->rect_params.top,
      obj->rect_params.width, obj->rect_params.height, obj->confidence,
      suffix);
  f->debug_logged++;
}

static GstPadProbeReturn
exclusion_probe (GstPad * pad, GstPadProbeInfo * info, gpointer user_data)
{
  ExclusionFilter *f = user_data;
  NvDsBatchMeta *batch =
      gst_buffer_get_nvds_batch_meta (GST_PAD_PROBE_INFO_BUFFER (info));
  if (!batch)
    return GST_PAD_PROBE_OK;

  for (NvDsMetaList * lf = batch->frame_meta_list; lf; lf = lf->next) {
    NvDsFrameMeta *frame = lf->data;
    const GArray *rects =
        mv3dt_exclusion_rules_for_source (f->rules, frame->source_id);
    if (!rects)
      continue;
    NvDsMetaList *lo = frame->obj_meta_list;
    while (lo) {
      NvDsObjectMeta *obj = lo->data;
      lo = lo->next;            /* advance before a removal unlinks the node */
      const NvOSD_RectParams *b = &obj->rect_params;
      if (mv3dt_exclusion_rects_inside_any (rects, b->left, b->top,
              b->width, b->height)) {
        debug_box (f, "dropped", frame->source_id, frame->frame_num, obj, "");
        nvds_remove_obj_meta_from_frame (frame, obj);
        f->dropped[frame->source_id]++;
      } else if (f->debug && f->debug_logged < MV3DT_DEBUG_MAX_LINES
          && mv3dt_exclusion_rects_overlap_any (rects, b->left, b->top,
              b->width, b->height)) {
        debug_box (f, "kept", frame->source_id, frame->frame_num, obj,
            " (overlaps, not inside)");
      }
    }
  }

  gint64 now = g_get_monotonic_time ();
  if (now - f->last_log_us >= MV3DT_LOG_INTERVAL_US) {
    for (guint s = 0; s < MV3DT_EXCLUSION_MAX_SOURCES; s++)
      if (f->rules->rects[s])
        g_print ("mv3dt-exclusion: source %u dropped %" G_GUINT64_FORMAT
            " boxes so far\n", s, f->dropped[s]);
    f->last_log_us = now;
  }
  return GST_PAD_PROBE_OK;
}

gboolean
mv3dt_exclusion_attach (GstElement * primary_gie_bin)
{
  const gchar *path = g_getenv ("MV3DT_EXCLUSION_CONFIG");
  if (!path || !*path) {
    g_print ("mv3dt-exclusion: MV3DT_EXCLUSION_CONFIG not set; "
        "filter disabled\n");
    return TRUE;
  }
  if (!primary_gie_bin) {
    g_printerr ("mv3dt-exclusion: no primary detector; filter disabled\n");
    return TRUE;
  }
  ExclusionFilter *f = load_filter (path);
  if (!f)
    return TRUE;
  GstPad *pad = gst_element_get_static_pad (primary_gie_bin, "src");
  if (!pad) {
    g_printerr ("mv3dt-exclusion: detector has no src pad; filter disabled\n");
    filter_free (f);
    return TRUE;
  }
  gst_pad_add_probe (pad, GST_PAD_PROBE_TYPE_BUFFER, exclusion_probe, f,
      filter_free);
  gst_object_unref (pad);
  return TRUE;
}
