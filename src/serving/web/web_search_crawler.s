module web_search_crawler
struct web_search_config {
    search_engines: list<string> = ["google", "bing"]
    max_results_per_engine: int = 10
    total_max_results: int = 20
    string google_api_key
    string google_cx_id
    string bing_api_key
    language: string = "zh-CN"
    region: string = "CN"
    safe_search: string = "moderate"
    time_range: string = ""
    enable_crawling: bool = true
    crawl_timeout_seconds: int = 15
    max_pages_to_crawl: int = 5
    max_content_length: int = 50000
    extract_main_content: bool = true
    follow_redirects: bool = true
    user_agent: string = "NEURX-Bot/1.0 (Research Crawler; +https:
    remove_scripts_styles: bool = true
    normalize_whitespace: bool = true
    extract_metadata: bool = true
    extract_images_info: bool = false
    extract_links: bool = false
    deduplicate_results: bool = true
    rerank_with_llm: bool = false
    result_summary_length: int = 200
    cache_enabled: bool = true
    cache_ttl_hours: int = 24
}

struct search_result_item {
    string url
    string title
    string snippet
    string source_engine
    int rank_in_engine
    float relevance_score
    string published_date
    crawled_content crawled_content
    string crawl_status
    string crawl_error
}

struct crawled_content {
    int raw_html_size
    string text_content
    string cleaned_text
    page_metadata metadata
    list<page_section> sections
    float extraction_timestamp
    int word_count
}

struct page_metadata {
    string title
    string description
    string author
    string publish_date
    string last_modified
    string site_name
    string domain
    string language
    string content_type
    string canonical_url
    list<string> keywords
    og_data: map<string, string>
}

struct page_section {
    string heading
    int level
    string content
}

struct search_response {
    string query
    string corrected_query
    int total_results_found
    list<search_result_item> results
    list<string> aggregated_from_engines
    float search_time_ms
    string summary
    list<string> key_findings
    search_statistics stats
}

struct search_statistics {
    engine_query_times_ms: map<string, float>
    list<float> crawl_times_ms
    float total_crawl_time_ms
    int pages_crawled
    int pages_failed
    int duplicates_removed
    int cache_hit_count
}
interface search_engine_interface {
    name: string { get }
    search(string query, config: web_search_config)
}

struct engine_search_result {
    list<search_result_item> items
    int total_estimated
    string corrected_query
    float query_time_ms
    bool has_more
    string error
}

struct google_search_engine implements search_engine_interface {
    name = "google"
    string api_key
    string cx_id
    init(api_key: string, cx_id: string) {
        this.api_key = api_key
        this.cx_id = cx_id
    }
    get name . string {
        return this.name
    }
    search(string query, config: web_search_config) {
        start_time = current_time_millis()
        if this.api_key != null && this.cx_id != null {
            result = this._search_via_api(query, config, start_time)
        } else {
            result = this._search_public(query, config, start_time)
        }
        return result
    }
    _search_via_api(string query, config: web_search_config, float start_time) {
        params = {
            "key": this.api_key!,
            "cx": this.cx_id!,
            "q": query,
            "num": min(config.max_results_per_engine, 10),
            "hl": config.language.split("-")[0],
            "gl": config.region
        }
        if !config.time_range.empty():
            params["date_restrict"] = config.time_range
        response = http_get("https:
        data = json_decode(response.body)
        if "error" in data:
            return engine_search_result{
                items=[], total_estimated=0, query_time_ms=current_time_millis() - start_time,
                has_more=false, error=data["error"]["message"]
            }
        items: list<search_result_item> = []
        for i, item in enumerate(data.get("items", [])) {
            items.append(search_result_item{
                url=item["link"],
                title=item.get("title", ""),
                snippet=item.get("snippet", ""),
                source_engine=this.name,
                rank_in_engine=i + 1,
                published_date=item.get("pagemap", {}).get("metatags", [{}])[0].get("article:published_time")
            })
        return engine_search_result{
            items=items,
            total_estimated=data.get("searchInformation", {}).get("totalResults", 0),
            query_time_ms=current_time_millis() - start_time,
            has_more=False
        }
    }
    _search_public(string query, config: web_search_config, float start_time) {
        try {
            ddg_result = this._duckduckgo_fallback(query, config)
            for item in ddg_result.items:
                item.source_engine = this.name
            return ddg_result
        } catch exception as e:
            return engine_search_result{
                items=[], total_estimated=0, query_time_ms=current_time_millis() - start_time,
                has_more=false, error=f"Google search failed: {e.message}"
            }
    }
    _duckduckgo_fallback(string query, config: web_search_config) {
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1"
        }
        response = http_get("https:
        data = json_decode(response.body)
        items: list<search_result_item> = []
        if "abstract" in data and not data["abstract"].empty():
            items.append(search_result_item{
                url=data.get("abstract_url", ""),
                title=data.get("heading", ""),
                snippet=data["abstract"],
                source_engine="duck_duck_go",
                rank_in_engine=1
            })
        return engine_search_result{
            items=items,
            total_estimated=len(items),
            query_time_ms=response.elapsed * 1000,
            has_more=false
        }
    }
}

struct bing_search_engine implements search_engine_interface {
    name = "bing"
    string api_key
    init(api_key: string) {
        this.api_key = api_key
    }
    get name . string {
        return this.name
    }
    search(string query, config: web_search_config) {
        start_time = current_time_millis()
        if this.api_key != null:
            return this._search_via_api(query, config, start_time)
        else:
            return this._search_fallback(query, config, start_time)
    }
    _search_via_api(string query, config: web_search_config, float start_time) {
        headers = {
            "ocp-apim-subscription-key": this.api_key!
        }
        params = {
            "q": query,
            "count": min(config.max_results_per_engine, 50),
            "offset": 0,
            "mkt": f"{config.language}-{config.region}",
            "safesearch": config.safe_search
        }
        if !config.time_range.empty():
            params["freshness"] = match config.time_range {
                "d" => "day"
                "w" => "week"
                "m" => "month"
                _ => null
            }
        response = http_get(
            "https:
            headers=headers,
            params=params
        )
        data = json_decode(response.body)
        items: list<search_result_item> = []
        for i, item in enumerate(data.get("webPages", {}).get("value", [])):
            items.append(search_result_item{
                url=item["url"],
                title=item.get("name", ""),
                snippet=item.get("snippet", ""),
                source_engine=this.name,
                rank_in_engine=i + 1,
                published_date=item.get("dateLastCrawled")
            })
        return engine_search_result{
            items=items,
            total_estimated=data.get("webPages", {}).get("totalEstimatedMatches", 0),
            query_time_ms=current_time_millis() - start_time,
            has_more=false
        }
    }
    _search_fallback(string query, config: web_search_config, float start_time) {
        return engine_search_result{
            items=[],
            total_estimated=0,
            query_time_ms=current_time_millis() - start_time,
            has_more=false,
            error="Bing API key not provided"
        }
    }
}

struct web_crawler {
    web_search_config config
    HTTPSession session
    cache: LRUCache<string, crawled_content>
    MainContentExtractor content_extractor
    HTMLCleaner html_cleaner
    init(config: web_search_config) {
        this.config = config
        this.session = new http_session(
            timeout=config.crawl_timeout_seconds,
            user_agent=config.user_agent,
            allow_redirects=config.follow_redirects
        )
        this.cache = new lru_cache(capacity=1000)
        this.content_extractor = new main_content_extractor(config=config)
        this.html_cleaner = new html_cleaner(config=config)
    }
    async crawl(string url) {
        if this.config.cache_enabled && url in this.cache {
            cached = this.cache[url]
            if !this._is_cache_expired(cached):
                return cached, null
        }
        try {
            response = await this.session.get(url)
            if response.status_code != 200:
                return null, f"HTTP {response.status_code}"
            html_content = response.text
            if len(html_content) > this.config.max_content_length * 5:
                html_content = html_content[:this.config.max_content_length * 5]
            extracted = this.content_extractor.extract(html_content, url)
            cleaned_text = this.html_cleaner.clean(extracted.text_content)
            crawled = crawled_content{
                raw_html_size=len(html_content.encode('utf-8')),
                text_content=extracted.text_content,
                cleaned_text=cleaned_text,
                metadata=extracted.metadata,
                sections=extracted.sections,
                extraction_timestamp=current_timestamp(),
                word_count=len(cleaned_text.split())
            }
            if this.config.cache_enabled:
                this.cache[url] = crawled
            return crawled, null
        } except timeout_exception:
            return null, "Timeout"
        except ssl_error:
            return null, "SSL certificate error"
        except dns_exception:
            return null, "DNS resolution failed"
        except too_many_redirects:
            return null, "Too many redirects"
        catch exception as e:
            return null, str(e)
    }
    batch_crawl(urls: list<string>, int max_concurrent = 3) {
        results: dict<string, tuple<crawled_content, string>> = {}
        semaphore = semaphore(max_concurrent)
        async def crawl_single(string url) {
            async with semaphore:
                results[url] = await this.crawl(url)
        tasks = [crawl_single(url) for url in urls]
        run_concurrently(tasks)
        return results
    }
    _is_cache_expired(cached: crawled_content) {
        age_hours = (current_timestamp() - cached.extraction_timestamp) / 3600
        return age_hours > this.config.cache_ttl_hours
    }
}

struct main_content_extractor {
    web_search_config config
    init(config: web_search_config) {
        this.config = config
    }
    extract(string html, string base_url) {
        soup = parse_html(html)
        self._remove_unwanted(soup)
        metadata = self._extract_metadata(soup, base_url)
        main_element = this._find_main_content(soup)
        if main_element != None:
            text_content, sections = this._extract_structured(main_element)
        else:
            body = soup.find("body")  soup
            text_content, sections = this._extract_structured(body)
        return extraction_result{
            text_content=text_content,
            metadata=metadata,
            sections=sections
        }
    }
    _remove_unwanted(soup: any) {
        tags_to_remove = ["script", "style", "noscript", "iframe", "nav", "footer",
                          "header", "aside", "form", ".advertisement", ".ads",
                          ".sidebar", ".menu", ".navigation", ".comment"]
        for selector in tags_to_remove:
            if selector.startswith("."):
                for elem in soup.find_all(class_=selector[1:]):
                    elem.decompose()
            elif selector.startsWith("#"):
                elem = soup.find(id=selector[1:])
                if elem != None { elem.decompose() }
            else:
                for elem in soup.find_all(selector):
                    elem.decompose()
    }
    _extract_metadata(soup: any, string base_url) {
        meta = page_metadata{
            title=soup.title.string.trim() if soup.title else "",
            site_name="",
            domain=extract_domain(base_url),
            canonical_url=base_url
        }
        desc_tag = soup.find("meta", attrs={"name": "description"})
        if desc_tag != None:
            meta.description = desc_tag.get("content", "").strip()
        author_tag = soup.find("meta", attrs={"name": "author"})
        if author_tag != None:
            meta.author = author_tag.get("content", "").strip()
        date_tag = soup.find("meta", attrs={"property": "article:published_time"})
        if date_tag == None:
            date_tag = soup.find("meta", attrs={"name": "date"})
        if date_tag != None:
            meta.publish_date = date_tag.get("content", "")
        og_data: dict<string, string> = {}
        og_tags = soup.find_all("meta", property=re.compile(r'^og:'))
        for tag in og_tags:
            prop = tag.get("property", "")
            content = tag.get("content", "")
            if !prop.empty() and !content.empty():
                og_data[prop] = content
        if og_data.length > 0:
            meta.og_data = og_data
            if "og:title" in og_data:
                meta.title = og_data["og:title"]
            if "og:site_name" in og_data:
                meta.site_name = og_data["og:site_name"]
        canonical = soup.find("link", rel="canonical")
        if canonical != None:
            meta.canonical_url = resolve_url(base_url, canonical.get("href", ""))
        html_tag = soup.find("html")
        if html_tag != None:
            meta.language = html_tag.get("lang", "")
        return meta
    }
    _find_main_content(soup: any) {
        candidates: list<{element: any, score: float}> = []
        for elem in soup.find_all(["div", "article", "main", "section"]):
            score = 0.0
            text_len = len(elem.get_text())
            if text_len < 150:
                continue
            tag = elem.name
            if tag == "article":
                score += 25
            if tag == "main":
                score += 20
            class_id_str = (elem.get("class", [])  []).join("") + (elem.get("id", "")  "")
            positive_keywords = ["post", "article", "content", "entry", "body", "text", "story",
                                  "main-content", "content-area", "article-body"]
            for kw in positive_keywords:
                if kw in class_id_str.lower():
                    score += 15
            link_elements = elem.find_all("a")
            text = elem.get_text()
            if len(text) > 0 and len(link_elements) > 0:
                link_density = sum(len(a.get_text()) for a in link_elements) / len(text)
                if link_density < 0.3:
                    score += 20
                elif link_density > 0.6:
                    score -= 25
            score += math.log(max(text_len, 1))
            negative_keywords = ["comment", "sidebar", "footer", "widget", "ad-", "promo",
                                  "related", "sponsor", "newsletter", "signup", "subscribe"]
            for kw in negative_keywords:
                if kw in class_id_str.lower():
                    score -= 30
            if score > 0:
                candidates.append({element=elem, score=score})
        if candidates.length > 0:
            candidates.sort_by_descending(c => c.score)
            return candidates[0].element
        return None
    }
    _extract_structured(element: any) {
        sections: list<page_section> = []
        content_parts: list<string> = []
        def process_node(node: any, depth: int = 0) {
            if node.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                level = int(node.name[1])
                text = node.get_text().strip()
                sections.append(page_section{heading=text, level=level, content=""})
                content_parts.append("\n" + "#" * level + " " + text + "\n")
            elif node.name == "p":
                text = clean_whitespace(node.get_text())
                if !text.empty():
                    sections.append(page_section{heading=null, level=0, content=text})
                    content_parts.append(text + "\n\n")
            elif node.name in ["ul", "ol"]:
                items: list<string> = []
                for li in node.find_all("li", recursive=False):
                    item_text = clean_whitespace(li.get_text())
                    if !item_text.empty():
                        items.append(item_text)
                marker = "-" if node.name == "ul" else "1."
                list_text = "\n".join(f"{marker} {item}" for item in items) + "\n\n"
                sections.append(page_section{heading=null, level=0, content=list_text.strip()})
                content_parts.append(list_text)
            elif node.name == "blockquote":
                quote_text = node.get_text().strip()
                sections.append(page_section{heading=null, level=0, content=f"> {quote_text}"})
                content_parts.append(f"\n> {quote_text}\n\n")
            for child in node.children:
                process_node(child, depth + 1)
        process_node(element)
        full_text = "".join(content_parts).strip()
        return full_text, sections
    }

    struct extraction_result {
        string text_content
        page_metadata metadata
        list<page_section> sections
    }
}

struct html_cleaner {
    web_search_config config
    init(config: web_search_config) {
        this.config = config
    }
    clean(string raw_text) {
        text = raw_text
        if this.config.remove_scripts_styles:
            text = regex.sub(r'<script[^>]*>.*</script>', '', text, flags=regex.DOTALL)
            text = regex.sub(r'<style[^>]*>.*</style>', '', text, flags=regex.DOTALL)
        text = regex.sub(r'<[^>]+>', ' ', text)
        if this.config.normalize_whitespace:
            text = regex.sub(r'[ \t]+', ' ', text)
            text = regex.sub(r'\n\s*\n+', '\n\n', text)
            text = regex.sub(r'\n{3,}', '\n\n', text)
            text = text.strip()
        text = unescape(text)
        return text
    }
}

struct web_search_system {
    web_search_config config
    engines: map<string, search_engine_interface>
    WebCrawler crawler
    ResultAggregator result_aggregator
    any llm_client
    init(config: web_search_config, llm_client: any) {
        this.config = config  new web_search_config()
        this.engines = map<string, search_engine_interface>{}
        this.llm_client = llm_client
        if "google" in this.config.search_engines:
            google = new google_search_engine(this.config.google_api_key, this.config.google_cx_id)
            this.engines["google"] = google
        if "bing" in this.config.search_engines:
            bing = new bing_search_engine(this.config.bing_api_key)
            this.engines["bing"] = bing
        this.crawler = new web_crawler(config=this.config)
        this.result_aggregator = new result_aggregator(config=config)
    }
    async search(string query, options: search_options) {
        opts = options  new search_options()
        start_total = current_time_millis()
        print(f"🔍 Searching: {query}")
        all_items: list<search_result_item> = []
        engine_times: map<string, float> = {}
        used_engines: list<string> = []
        tasks: list<tuple<string, ()
        for eng_name, engine in this.engines {
            tasks.append((eng_name, () => engine.search(query, this.config)))
        }
        engine_results = await run_tasks_concurrently(tasks, max_workers=len(this.engines))
        for eng_name, result in engine_results:
            engine_times[eng_name] = result.query_time_ms
            if result.error == None {
                all_items.extend(result.items)
                used_engines.append(eng_name)
                if result.corrected_query != None:
                    opts.corrected_query = result.corrected_query!
            if opts.verbose:
                status_icon = "✅" if result.error == None else "❌"
                print(f"   {status_icon} {eng_name}: {len(result.items)} results ({result.query_time_ms:.0f}ms)")
        if this.config.deduplicate_results and all_items.length > 0:
            before_dedup = len(all_items)
            all_items = this.result_aggregator.deduplicate(all_items)
            deduped_count = before_dedup - len(all_items)
            if opts.verbose and deduped_count > 0:
                print(f"   🗑️ Removed {deduped_count} duplicate results")
        if all_items.length > 0:
            all_items = this.result_aggregator.score_and_rank(all_items, query)
        if len(all_items) > this.config.total_max_results:
            all_items = all_items[:this.config.total_max_results]
        crawl_times: list<float> = []
        crawl_success = 0
        crawl_fail = 0
        if this.config.enable_crawling and opts.crawl_results:
            urls_to_crawl = [item.url for item in all_items[:min(this.config.max_pages_to_crawl, len(all_items))]]
            if opts.verbose:
                print(f"   🕷️ Crawling {len(urls_to_crawl)} pages...")
            crawl_start = current_time_millis()
            crawl_results = await this.crawler.batch_crawl(urls_to_crawl, max_concurrent=3)
            crawl_total_time = current_time_millis() - crawl_start
            for i, item in enumerate(all_items):
                if item.url in crawl_results:
                    content, error = crawl_results[item.url]
                    if content != None:
                        item.crawled_content = content
                        item.crawl_status = "success"
                        crawl_success += 1
                    else:
                        item.crawl_status = "failed"
                        item.crawl_error = error
                        crawl_fail += 1
                    item.rank_in_engine = i + 1
                    crawl_times.append(crawl_total_time / max(len(urls_to_crawl), 1))
        summary = None
        key_findings: list<string> = None
        if this.llm_client != None and (opts.generate_summary or this.config.rerank_with_llm):
            if opts.verbose:
                print(f"   🤖 Generating AI summary...")
            context = this._build_context_for_llm(query, all_items[:5])
            summary_result = this._generate_summary_with_llm(query, context)
            summary = summary_result.summary
            key_findings = summary_result.key_findings
            if this.config.rerank_with_llm and summary_result.reranked_indices != None:
                all_items = [all_items[i] for i in summary_result.reranked_indices!]
        total_time = current_time_millis() - start_total
        if opts.verbose:
            print(f"\n   📊 Results: {len(all_items)} items in {total_time:.0f}ms")
            print(f"      Engines: {', '.join(used_engines)}")
            if crawl_success + crawl_fail > 0:
                print(f"      Crawled: {crawl_success} success, {crawl_fail} failed")
        return search_response{
            query=query,
            corrected_query=opts.corrected_query,
            total_results_found=len(all_items),
            results=all_items,
            aggregated_from_engines=used_engines,
            search_time_ms=total_time,
            summary=summary,
            key_findings=key_findings,
            stats=search_statistics{
                engine_query_times_ms=engine_times,
                crawl_times_ms=crawl_times,
                total_crawl_time_ms=sum(crawl_times) if crawl_times.length > 0 else 0,
                pages_crawled=crawl_success,
                pages_failed=crawl_fail,
                duplicates_removed=(before_dedup - len(all_items)) if this.config.deduplicate_results else 0,
                cache_hit_count=0
            }
        }
    }
    _build_context_for_llm(string query, top_results: list<search_result_item>) {
        parts: list<string> = []
        parts.append(f"Query: {query}\n")
        parts.append("=" * 60 + "\n")
        for i, result in enumerate(top_results):
            parts.append(f"[{i+1}] {result.title}")
            parts.append(f"    URL: {result.url}")
            if result.snippet != None and !result.snippet!.empty():
                parts.append(f"    Snippet: {result.snippet!}")
            if result.crawled_content != None:
                preview = result.crawled_content!.cleaned_text[:500]
                if len(result.crawled_content!.cleaned_text) > 500:
                    preview += "... [truncated]"
                parts.append(f"    Content Preview:\n    {preview}")
            parts.append("")
        return "\n".join(parts)
    }
    _generate_summary_with_llm(string query, string context) {
        prompt = f"""
Based on the following search results for the query "{query}", provide:
1. A concise summary (3-4 sentences) answering the query
2. 3-5 key findings or important points
Search Results:
{context}
Respond in this format:
[your summary here]
- [finding 1]
- [finding 2]
- [finding 3]
Also provide a comma-separated ranking of the most relevant result indices (0-based): [indices]
"""
        response = this.llm_client!.generate(prompt, temperature=0.3, max_tokens=600)
        summary_match = regex.search(r'## Summary\n(.*)(=##|\Z)', response.text, regex.DOTALL)
        findings_match = regex.search(r'## Key Findings\n(.*)(=\Z)', response.text, regex.DOTALL)
        indices_match = regex.search(r'indices:\s*\[(.*)\]', response.text)
        summary = summary_match.group(1).trim() if summary_match else ""
        key_findings: list<string> = []
        if findings_match != None:
            findings_text = findings_match.group(1)
            key_findings = [line.strip().lstrip("- ") for line in findings_text.split("\n")
                           if line.trim().starts_with("-")]
        reranked_indices: list<int> = None
        if indices_match != None:
            indices_str = indices_match.group(1)
            try:
                reranked_indices = [int(x.trim()) for x in indices_str.split(",")]
            }
        return llm_summary_result{
            summary=summary,
            key_findings=key_findings if key_findings.length > 0 else None,
            reranked_indices=reranked_indices
        }
    }

    struct llm_summary_result {
        string summary
        list<string> key_findings
        list<int> reranked_indices
    }
    export_results(response: search_response, string format = "markdown", output_path: string) {
        output: list<string> = []
        output.append(f"# Search Results: {response.query}\n")
        output.append(f"**Total Results**: {response.total_results_found}\n")
        output.append(f"**Time**: {response.search_time_ms:.0f}ms\n")
        output.append(f"**Sources**: {', '.join(response.aggregated_from_engines)}\n")
        if response.corrected_query != None:
            output.append(f"**Corrected Query**: {response.corrected_query!}\n")
        if response.summary != None:
            output.append(f"---\n\n## Summary\n\n{response.summary!}\n")
        if response.key_findings != None:
            output.append("## Key Findings\n")
            for finding in response.key_findings!:
                output.append(f"- {finding}\n")
            output.append("")
        output.append("---\n\n## Detailed Results\n")
        for i, item in enumerate(response.results, 1):
            output.append(f"### {i}. {item.title}\n")
            output.append(f"- **URL**: [{item.url}]({item.url})")
            output.append(f"- **Source**: {item.source_engine} (Rank: #{item.rank_in_engine})")
            if item.published_date != None:
                output.append(f"- **Date**: {item.published_date!}")
            if item.snippet != None and !item.snippet!.empty():
                output.append(f"\n**Snippet**:\n> {item.snippet!}\n")
            if item.crawled_content != None and item.crawl_status == "success":
                cc = item.crawled_content!
                output.append(f"**Content** ({cc.word_count} words):\n")
                preview = cc.cleaned_text[:1000]
                if len(cc.cleaned_text) > 1000:
                    preview += "\n... [truncated]"
                output.append(preview + "\n")
                if cc.metadata.author != None:
                    output.append(f"*Author: {cc.metadata.author!}*\n")
            output.append("---\n")
        final_output = "\n".join(output)
        if output_path != None:
            write_file(output_path!, final_output)
            print(f"Results saved to: {output_path!}")
        return final_output
    }
}

struct search_options {
    crawl_results: bool = true
    generate_summary: bool = true
    verbose: bool = true
    string corrected_query
}

struct result_aggregator {
    web_search_config config
    init(config: web_search_config) {
        this.config = config
    }
    deduplicate(items: list<search_result_item>) {
        seen_urls: set<string> = set{}
        unique_items: list<search_result_item> = []
        for item in items:
            normalized = self._normalize_url(item.url)
            if normalized not in seen_urls:
                seen_urls.add(normalized)
                unique_items.append(item)
        return unique_items
    }
    score_and_rank(items: list<search_result_item>, string query) {
        scored_items: list<tuple<search_result_item, float>> = []
        query_terms = set(query.to_lower().split_whitespace())
        for item in items {
            score = 0.0
            title_lower = item.title.to_lower()
            title_matches = sum(1 for term in query_terms if term in title_lower)
            score += title_matches * 15.0
            snippet_lower = (item.snippet  "").to_lower()
            snippet_matches = sum(1 for term in query_terms if term in snippet_lower)
            score += snippet_matches * 8.0
            score += (20.0 / (item.rank_in_engine + 1))
            domain = extract_domain(item.url)
            if this._is_high_quality_domain(domain):
                score += 5.0
            if item.published_date != None:
                days_old = days_since(item.published_date!)
                if days_old < 30:
                    score += 10.0
                elif days_old < 180:
                    score += 5.0
            if len(item.url) > 120:
                score -= 3.0
            item.relevance_score = round(score, 2)
            scored_items.append((item, score))
        }
        scored_items.sort_by_descending(x => x[1])
        return [item for item, _ in scored_items]
    }
    _normalize_url(string url) {
        parsed = urlparse(url)
        normalized = f"{parsed.scheme}:
        if parsed.query:
            params = sorted(parsed.query.split("&"))
            normalized += "" + "&".join(params)
        return normalized.to_lower()
    }
    _is_high_quality_domain(string domain) {
        quality_indicators = [
            ".edu", ".gov", ".org",
            "wikipedia.org", "arxiv.org", "github.com",
            "stackoverflow.com", "medium.com", "nature.com",
            "science.org", "ieee.org", "acm.org"
        ]
        return any(indicator in domain for indicator in quality_indicators)
    }
}

function create_web_search_system(config: web_search_config, llm_client: any) {
    return new WebSearchSystem(config=config, llm_client=llm_client)
}
async function test_web_search_system() {
    print("🧪 testing NEURX WEB search system...")
    cfg = web_search_config(
        enable_crawling=false,
        max_results_per_engine=3,
        total_max_results=5
    )
    ws = create_web_search_system(cfg)
    print("  ✓ test 1: System initialization & engine setup")
    assert len(ws.engines) >= 1, "at least one search engine should be initialized"
    assert ws.crawler != None, "crawler should be initialized"
    print("  ✓ test 2: URL normalization & deduplication")
    agg = ws.result_aggregator
    test_items = [
        search_result_item{url="https:
        search_result_item{url="HTTPS:
        search_result_item{url="https:
        search_result_item{url="https:
    ]
    unique = agg.deduplicate(test_items)
    assert len(unique) == 3, f"dedup failed: expected 3, got {len(unique)}"
    print("  ✓ test 3: Result scoring & ranking")
    scored = agg.score_and_rank(unique, "test query example")
    assert len(scored) == len(unique), "scoring should preserve all items"
    assert scored[0].relevance_score != None, "relevance scores should be assigned"
    for i in range(len(scored) - 1):
        assert scored[i].relevance_score! >= scored[i+1].relevance_score!, \
               f"results should be sorted by score: {scored[i].relevance_score} >= {scored[i+1].relevance_score}"
    print("  ✓ test 4: HTML cleaning")
    cleaner = new HTMLCleaner(cfg)
    dirty_html = "<div><script>alert('xss')</script><p>Hello   World</p></div>"
    cleaned = cleaner.clean(dirty_html)
    assert "<script" not in cleaned, "Script tags should be removed"
        assert "Hello   World" in cleaned or "Hello World" in cleaned, "Text content preserved"
    print("\n✅ All Web Search System Tests Passed!")
    return true
}
export {
    web_search_config, search_result_item, crawled_content, page_metadata, page_section,
    search_response, search_statistics,
    web_search_system, search_options,
    create_web_search_system, test_web_search_system
}
