import argparse
import asyncio
import aiohttp
import time
from collections import defaultdict
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import requests
import json
import aiohttp
from datetime import datetime, timedelta
import asyncio
import threading
import settings
from libs.logger import setup_logger


class GenAIClient:
    """Thread-safe singleton client for GenAI Retrieval API."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        if hasattr(self, "_initialized"):
            return

        self.token_url = settings.GEN_AI_RETRIEVAL_TOKEN_URL
        self.client_id = settings.GEN_AI_RETRIEVAL_CLIENT_ID
        self.service_account = settings.GEN_AI_RETRIEVAL_SERVICE_ACCOUNT
        self.password = settings.GEN_AI_RETRIEVAL_PASSWORD
        self.api_endpoint = (
            f"{settings.GEN_AI_RETRIEVAL_ENDPOINT}{settings.GEN_AI_RETRIEVAL_PATH}"
        )
        self.logger = setup_logger(__name__)

        self._cached_jwt_token = None
        self._token_expiry = None
        self._token_lock = threading.Lock()
        self._initialized = True

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of GenAIClient (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _validate_credentials(self):
        """Check if required credentials are present."""
        missing = []
        if not self.client_id:
            missing.append("GEN_AI_RETRIEVAL_CLIENT_ID")
        if not self.service_account:
            missing.append("GEN_AI_RETRIEVAL_SERVICE_ACCOUNT")
        if not self.password:
            missing.append("GEN_AI_RETRIEVAL_PASSWORD")

        if missing:
            self.logger.error(f"Missing credentials: {', '.join(missing)}")
            return False
        return True

    def _make_token_request(self, request_data):
        """Make a token request and return the response JSON."""
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(
                self.token_url, json=request_data, headers=headers, timeout=15
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Token request failed: {e}")
            return None

    def get_auth_token(self):
        """Get authentication token from the API."""
        if not self._validate_credentials():
            return None

        request_data = {
            "client_id": self.client_id,
            "grant_type": "password",
            "username": self.service_account,
            "password": self.password,
            "scope": "openid service_account_id",
            "connection": "service-account",
        }

        response_json = self._make_token_request(request_data)
        if not response_json:
            return None

        id_token = response_json.get("id_token")
        if not id_token:
            self.logger.error("Auth token 'id_token' not found in response")
        return id_token

    def _get_jwt_token(self):
        """Get JWT token with caching (60s buffer before expiry)."""
        # Check cache
        with self._token_lock:
            if self._cached_jwt_token and self._token_expiry:
                if datetime.now() < self._token_expiry - timedelta(seconds=60):
                    remaining = (self._token_expiry - datetime.now()).total_seconds()
                    self.logger.debug("Using cached JWT token")
                    return self._cached_jwt_token, int(remaining)

        # Fetch new token
        auth_token = self.get_auth_token()
        if not auth_token:
            self.logger.error("Failed to get auth token")
            return None, 0

        request_data = {
            "client_id": self.client_id,
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "scope": "openid pib",
            "assertion": auth_token,
        }

        response_json = self._make_token_request(request_data)
        if not response_json:
            return None, 0

        jwt_token = response_json.get("access_token")
        expires_in = response_json.get("expires_in", 0)

        if not jwt_token:
            self.logger.error("JWT token 'access_token' not found in response")
            return None, 0

        # Update cache
        with self._token_lock:
            self._cached_jwt_token = jwt_token
            self._token_expiry = datetime.now() + timedelta(seconds=expires_in)
            self.logger.info(
                f"JWT token retrieved and cached, expires in {expires_in}s"
            )

        return jwt_token, expires_in

    def invalidate_token_cache(self):
        """Force token refresh on next request."""
        with self._token_lock:
            self._cached_jwt_token = None
            self._token_expiry = None
            self.logger.info("JWT token cache invalidated")

    def _validate_dates(self, init_date, end_date, thread_id):
        """Validate and return date strings, defaulting to last 90 days if invalid."""
        try:
            if init_date is None or end_date is None:
                raise ValueError("Date is None")
            datetime.strptime(init_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")
            return init_date, end_date
        except (ValueError, TypeError):
            self.logger.error(
                f"Thread {thread_id}: Invalid dates ({init_date}, {end_date}). "
                "Defaulting to last 90 days"
            )
            today = datetime.now()
            return (today - timedelta(days=90)).strftime("%Y-%m-%d"), today.strftime(
                "%Y-%m-%d"
            )

    def _build_request_data(
        self,
        query_text,
        response_limit,
        date_from,
        date_to,
        use_semantic_search=False,
        use_lexical=False,
    ):
        """Build the API request payload."""
        attributes = {
            "response_limit": response_limit,
            "query": {
                "search_filters": [{"scope": "Language", "value": "en"}],
                "value": query_text,
                "date": {"custom": {"from": date_from, "to": date_to}},
            },
        }

        # Add search_mode_override if semantic search is requested
        if use_semantic_search:
            attributes["search_mode_override"] = "Semantic"

        if use_lexical:
            attributes["search_mode_override"] = "Lexical"

        return {
            "data": {
                "attributes": attributes,
                "id": "GenAIRetrieval",
                "type": "genai-content",
            }
        }

    def _parse_results(self, response_json):
        """Parse API response into results list."""
        return [
            {
                "meta": item.get("meta", {}),
                "attributes": item.get("attributes", {}),
                "id": item.get("meta", {}).get("original_doc_id", f"no_id_{idx}"),
            }
            for idx, item in enumerate(response_json.get("data", []))
        ]

    async def genai_search_call(
        self,
        query_text,
        response_limit,
        init_date,
        end_date,
        session,
        thread_id,
        full_article=False,
        use_semantic_search=False,
        use_lexical=False,
    ):
        """
        Perform async GenAI API search with retry logic.

        Args:
            query_text: Text of the search query to send to the GenAI API.
            response_limit: Maximum number of results to return.
            init_date: Start date for the search window.
            end_date: End date for the search window.
            session: An aiohttp client session used to perform the HTTP request.
            thread_id: Identifier used for logging and tracing this call.
            full_article: If True, would retrieve full articles (currently not implemented).
            use_semantic_search: If True, uses semantic search mode. If False, uses default hybrid search.

        Returns:
            tuple: (results, call_audit_info, response_json)
        """
        if full_article:
            self.logger.error("Full article functionality not implemented")

        # Get JWT token
        jwt_token, _ = self._get_jwt_token()
        if not jwt_token:
            self.logger.error(
                f"Thread {thread_id}: Invalid JWT token. Skipping API call"
            )
            return [], {}, None

        # Validate dates
        date_from, date_to = self._validate_dates(init_date, end_date, thread_id)

        # Build request
        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.dowjones.genai-content.v_1.0",
        }
        data = self._build_request_data(
            query_text,
            response_limit,
            date_from,
            date_to,
            use_semantic_search,
            use_lexical,
        )

        # Retry loop
        max_retries = 3
        retry_delay = 20
        call_audit_info = {}

        search_mode = "Semantic" if use_semantic_search else "Default"
        for attempt in range(max_retries):
            self.logger.info(
                f"Thread {thread_id}: Attempt {attempt + 1}/{max_retries} "
                f"(Dates: {date_from} to {date_to}, Limit: {response_limit}, "
                f"Search Mode: {search_mode})"
            )

            start_time = datetime.now()

            try:
                timeout = aiohttp.ClientTimeout(total=120)
                async with session.post(
                    self.api_endpoint, json=data, headers=headers, timeout=timeout
                ) as response:
                    call_duration = (datetime.now() - start_time).total_seconds()
                    call_audit_info["call_duration_s"] = round(call_duration, 2)

                    response_text = await response.text()

                    if response.status == 200:
                        try:
                            response_json = json.loads(response_text)
                            if response_json.get("data"):
                                self.logger.info(
                                    f"Thread {thread_id}: Success on attempt {attempt + 1} "
                                    f"({call_duration:.2f}s)"
                                )
                                results = self._parse_results(response_json)
                                return results, call_audit_info, response_json
                            else:
                                self.logger.warning(
                                    f"Thread {thread_id}: Attempt {attempt + 1} returned no data"
                                )
                        except json.JSONDecodeError as e:
                            self.logger.error(
                                f"Thread {thread_id}: JSON decode failed on attempt {attempt + 1}: {e}"
                            )
                    else:
                        self.logger.error(
                            f"Thread {thread_id}: HTTP {response.status} on attempt {attempt + 1}. "
                            f"Response: {response_text}"
                        )
                        if response.status == 400:
                            self.logger.warning(
                                f"Thread {thread_id}: 400 Bad Request — skipping retries. "
                                f"Query: {query_text}, Dates: {date_from} to {date_to}, Limit: {response_limit}"
                            )
                            return [], call_audit_info, None

            except asyncio.TimeoutError:
                duration = (datetime.now() - start_time).total_seconds()
                call_audit_info["call_duration_s"] = round(duration, 2)
                self.logger.error(
                    f"Thread {thread_id}: Timeout on attempt {attempt + 1}"
                )
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                call_audit_info["call_duration_s"] = round(duration, 2)
                self.logger.error(
                    f"Thread {thread_id}: Error on attempt {attempt + 1}: {e}"
                )

            # Wait before retry (except on last attempt)
            if attempt < max_retries - 1:
                self.logger.info(
                    f"Thread {thread_id}: Waiting {retry_delay}s before retry"
                )
                await asyncio.sleep(retry_delay)

        self.logger.error(f"Thread {thread_id}: All {max_retries} attempts failed")
        return [], call_audit_info, None


# ── Configuration ────────────────────────────────────────────────────────────
QUERY_TEXT = "artificial intelligence"
RESPONSE_LIMIT = 20
DATE_FROM = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
OFFSETS = list(range(5))  # 0 … 30 days before today
NUM_CALLS = 10  # calls per offset


# ── Search runner for one offset ─────────────────────────────────────────────
async def run_searches_for_offset(
    offset: int,
    sleep_seconds: int,
    session: aiohttp.ClientSession,
) -> dict:
    client = GenAIClient.get_instance()
    date_to = (datetime.now() - timedelta(days=offset)).strftime("%Y-%m-%d")

    article_appearances: dict[str, list[int]] = defaultdict(list)

    for i in range(NUM_CALLS):
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

        results, audit, _ = await client.genai_search_call(
            query_text=QUERY_TEXT,
            response_limit=RESPONSE_LIMIT,
            init_date=DATE_FROM,
            end_date=date_to,
            session=session,
            thread_id=f"offset-{offset}-call-{i}",
        )

        for article in results:
            article_appearances[article["id"]].append(i + 1)

        duration = audit.get("call_duration_s", "?")
        print(
            f"    [offset={offset:>2}d | call {i + 1}/{NUM_CALLS}] "
            f"{len(results)} articles in {duration}s"
        )

    total = len(article_appearances)
    always_count = sum(1 for c in article_appearances.values() if len(c) == NUM_CALLS)
    pct_always = (always_count / total * 100) if total else 0.0

    print(
        f"     → consistency: {pct_always:.1f}%  "
        f"({always_count}/{total} articles in all {NUM_CALLS} calls)\n"
    )

    return {"offset": offset, "date_to": date_to, "pct_always": pct_always}


# ── Full sweep ────────────────────────────────────────────────────────────────
async def run_sweep(sleep_seconds: int) -> list[dict]:
    total_calls = len(OFFSETS) * NUM_CALLS
    print(
        f"\nSweep: offsets 0–30 | {NUM_CALLS} calls each | {total_calls} total calls\n"
    )

    all_metrics = []
    async with aiohttp.ClientSession() as session:
        for offset in OFFSETS:
            date_to = (datetime.now() - timedelta(days=offset)).strftime("%Y-%m-%d")
            print(f"  ── Offset {offset:>2}d  (end date: {date_to}) ──")
            metrics = await run_searches_for_offset(offset, sleep_seconds, session)
            all_metrics.append(metrics)

    return all_metrics


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_results(all_metrics: list[dict], output_path: str):
    offsets = [m["offset"] for m in all_metrics]
    pct_always = [m["pct_always"] for m in all_metrics]

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.fill_between(offsets, pct_always, alpha=0.15, color="#2196F3")
    ax.plot(
        offsets,
        pct_always,
        marker="o",
        color="#2196F3",
        linewidth=2,
        markersize=6,
        zorder=3,
    )

    # Annotate each point
    for x, y in zip(offsets, pct_always):
        ax.annotate(
            f"{y:.0f}%",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=7.5,
            color="#1565C0",
        )

    ax.set_xlabel("End-date offset (days before today)", fontsize=11)
    ax.set_ylabel("% articles seen in ALL calls", fontsize=11)
    ax.set_title(
        f"Search Consistency vs End-Date Offset\n"
        f'Query: "{QUERY_TEXT}" | {NUM_CALLS} calls per offset | offsets 0–30',
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(offsets)
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {output_path}")
    plt.show()


# ── Summary table ─────────────────────────────────────────────────────────────
def print_summary(all_metrics: list[dict]):
    print("\n" + "=" * 45)
    print("SWEEP SUMMARY")
    print("=" * 45)
    print(f"{'Offset':>7}  {'End date':>12}  {'Consistency':>12}")
    print("─" * 45)
    for m in all_metrics:
        print(f"  {m['offset']:>4}d  {m['date_to']:>12}  {m['pct_always']:>11.1f}%")
    print("=" * 45)


# ── Entry point ───────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(
        description="Consistency sweep over end-date offsets 0–30"
    )
    parser.add_argument(
        "--sleep", type=int, default=0, help="Seconds between calls (default: 0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="consistency_sweep.png",
        help="Output path for the plot (default: consistency_sweep.png)",
    )
    args = parser.parse_args()

    all_metrics = await run_sweep(sleep_seconds=args.sleep)
    print_summary(all_metrics)
    plot_results(all_metrics, output_path=args.output)


if __name__ == "__main__":
    asyncio.run(main())
