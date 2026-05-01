from traffic_anomaly.pipeline import TrafficAnomalyPipeline, build_arg_parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    display = not (args.no_display or args.batch)
    pipeline = TrafficAnomalyPipeline(
        config_path=args.config,
        max_frames=args.max_frames,
        display=display,
        source_mode=args.source_mode,
        source_override=args.source,
        tracker_config_override=args.tracker_config,
        skip_frames=args.skip_frames,
        device=args.device,
        appearance_model=args.appearance_model,
        detector=args.detector,
        detector_weights=args.detector_weights,
        save_evidence=not args.no_save_evidence,
        save_normal_sequences=not args.no_save_normal_sequences,
        save_tracklets=not args.no_save_tracklets,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
