/*
 * mv3dt_exclusion.h - per-source detector exclusion rectangles, applied on
 * the primary detector's src pad so nvtracker never sees the dropped boxes.
 */
#ifndef MV3DT_EXCLUSION_H
#define MV3DT_EXCLUSION_H

#include <gst/gst.h>

/* Attach the filter to the primary detector bin when $MV3DT_EXCLUSION_CONFIG
 * names a readable config; otherwise do nothing. Always returns TRUE: a bad
 * config disables the filter, it never stops the pipeline. */
gboolean mv3dt_exclusion_attach (GstElement * primary_gie_bin);

#endif
