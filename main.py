from traffic_anomaly.pipeline import TrafficAnomalyPipeline, build_arg_parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    display = not (args.no_display or args.batch)
    pipeline = TrafficAnomalyPipeline(
        config_path=args.config,
        max_frames=args.max_frames,
        display=display,
        source_override=args.source,
        skip_frames=args.skip_frames,
    )
    pipeline.run()


if __name__ == "__main__":
    main()

