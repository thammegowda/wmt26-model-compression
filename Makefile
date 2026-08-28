# WMT26 Model Compression — organizer tooling
#
# Run `make help` to list goals.

FINDINGS := findings-report

.DEFAULT_GOAL := help
.PHONY: help refresh

help:  ## List available goals
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "} {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

refresh:  ## Sync the findings-report submodule with its Overleaf remote (latest)
	git submodule sync $(FINDINGS)
	git submodule update --init --remote $(FINDINGS)
