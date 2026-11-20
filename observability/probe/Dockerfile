ARG BASE_IMAGE
FROM busybox:1.37.0 AS probe
FROM ${BASE_IMAGE}
COPY --from=probe /bin/busybox /busybox
