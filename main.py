from traffic_anomaly.pipeline import TrafficAnomalyPipeline, build_arg_parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    pipeline = TrafficAnomalyPipeline(
        config_path=args.config,
        max_frames=args.max_frames,
        display=not args.no_display,
        source_override=args.source,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
