"""The ZentroMart support agent: retrieval + tools + conversation memory (Steps 2 and 3)."""
import json
import re
from dataclasses import dataclass, field

from . import data
from .config import TOP_K
from .kb import Retriever
from .normalize import normalize
from .prompts import AGENT_PROMPT, TOOLS

MAX_TOOL_ROUNDS = 6
MAX_VERIFY_ATTEMPTS = 2  # first try + one retry (tools.py docstring)

HINGLISH_WORDS = {"hai", "hain", "kya", "nahi", "nahin", "mera", "meri", "mere", "mujhe", "aap", "aapka", "karo",
                  "karein", "kab", "kaise", "kitne", "kitna", "chahiye", "abhi", "kar", "hoga", "tha", "thi", "pe",
                  "mein", "ke", "ka", "ki", "baare", "haan", "kyun", "kyon", "diya", "kiya", "karna", "hoon", "raha",
                  "rahi", "wala", "wali", "batao", "bataiye", "chahta", "chahti", "paisa", "paise", "wapas", "kal",
                  "aaj", "bhi", "toh", "lekin", "agar", "kaun", "kahan", "kidhar", "yeh", "woh", "sakte", "milega"}


def _no_hindi(text: str) -> bool:
    """True when a reply has no Devanagari and not a single Hinglish marker word."""
    if re.search(r"[\u0900-\u097F]", text or ""):
        return False
    return not (set(re.findall(r"[a-z]+", (text or "").lower())) & HINGLISH_WORDS)


def reply_language(text: str) -> str:
    if re.search(r"[\u0900-\u097F]", text or ""):
        return "hi"
    words = re.findall(r"[a-z]+", (text or "").lower())
    hits = len(set(words) & HINGLISH_WORDS)
    return "hi-en" if hits >= 2 or (hits == 1 and len(words) <= 8) else "en"


AI_INTRO = {
    "en": "Hi, I'm ZentroMart's AI assistant. ",
    "hi-en": "Namaste, main ZentroMart ki AI assistant hoon. ",
    "hi": "नमस्ते, मैं ZentroMart की AI सहायक हूँ। ",
}


@dataclass
class CallState:
    turns: int = 0
    order_id: str = None
    verified_orders: dict = field(default_factory=dict)  # order_id -> record
    failed_verifications: int = 0
    locked: bool = False
    tickets: list = field(default_factory=list)
    lang: str = None

    def block(self) -> str:
        lines = [f"CALL STATE (turn {self.turns + 1})",
                 f"- You have {'already' if self.turns else 'NOT yet'} introduced yourself as the AI assistant."]
        if self.order_id:
            lines.append(f"- Order under discussion: {self.order_id}")
        for oid, rec in self.verified_orders.items():
            lines.append(f"- Caller VERIFIED for order {oid}. Record: {json.dumps(rec, ensure_ascii=False, default=str)}")
        if self.failed_verifications:
            lines.append(f"- Failed verification attempts: {self.failed_verifications}/{MAX_VERIFY_ATTEMPTS}")
        if self.locked:
            lines.append("- Verification LOCKED after two failures: share no order details; offer a human colleague.")
        for t in self.tickets:
            lines.append(f"- Ticket already created this call ({t.get('intent') or 'handover'}): {t['summary'][:160]}")
        return "\n".join(lines)


class Agent:
    def __init__(self, llm, retriever: Retriever = None, top_k: int = None):
        self.llm = llm
        self.kb = retriever or Retriever()
        self.top_k = top_k or TOP_K
        self.tools = data.provided_tools()
        self.reset()

    def reset(self):
        self.state = CallState()
        self.history = []  # compact: user text + final replies only
        self.last_turn = None

    # ------------------------------------------------------------------ tools
    def _lookup_order(self, order_id=None, verification_value=None, **_):
        oid = re.sub(r"\D", "", str(order_id or ""))
        if not oid:
            return {"error": "order_id missing; ask the customer for it"}
        self.state.order_id = oid
        if oid in self.state.verified_orders and not verification_value:
            return {"found": True, "verified": True, "note": "already verified this call",
                    **self.state.verified_orders[oid]}
        if self.state.locked:
            return {"locked": True, "message": "Two failed verification attempts. Do not retry or reveal anything. Offer to connect a human colleague (create_ticket)."}
        res = self.tools.lookup_order(oid, verification_value)
        if verification_value not in (None, ""):
            if res.get("verified"):
                rec = {k: v for k, v in res.items() if v is not None and k not in ("found", "verified")}
                self.state.verified_orders[oid] = rec
                res["evidence"] = [f"orders.csv:{oid}"]
            else:
                self.state.failed_verifications += 1
                left = MAX_VERIFY_ATTEMPTS - self.state.failed_verifications
                if left <= 0:
                    self.state.locked = True
                    res["message"] = ("Verification failed twice. Stop: share nothing about the order, "
                                      "apologise and offer a human colleague via create_ticket.")
                else:
                    res["message"] = (f"Verification failed. {left} retry left: ask the customer to re-check "
                                      "the order ID and give phone last 4 digits or delivery pincode.")
        return res

    def _create_ticket(self, summary="", order_id=None, reason=None, intent=None, **_):
        oid = re.sub(r"\D", "", str(order_id or self.state.order_id or "")) or None
        res = self.tools.create_ticket(summary, order_id=oid, reason=reason, intent=intent)
        if res.get("created"):
            self.state.tickets.append({"summary": summary, "intent": intent, "ticket_id": res["ticket_id"]})
        return res

    def _search_kb(self, query="", **_):
        hits = self.kb.search(query, k=4)
        self._turn_retrieved.extend(c.id for c, _ in hits)
        return {"sections": [c.render(1200) for c, _ in hits]}

    def _call_tool(self, name, args):
        fn = {
            "lookup_order": self._lookup_order,
            "create_ticket": self._create_ticket,
            "search_kb": self._search_kb,
            "lookup_product": lambda query="", **_: data.lookup_product(query),
            "check_pincode": lambda pincode="", **_: data.check_pincode(pincode),
            "date_calc": lambda **kw: data.date_calc(**{k: v for k, v in kw.items()
                                                       if k in ("start_date", "add_business_days_n", "add_calendar_days_n")}),
        }.get(name)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**args)
        except Exception as e:
            return {"error": f"{e.__class__.__name__}: {e}"}

    def _auto_order_tools(self, norm) -> list:
        """Deterministic order steps: look up a newly given order ID, and verify when the caller
        gives phone last-4 / pincode while verification is pending. The LLM words the reply."""
        ents, out = norm["entities"], []
        for oid in ents.get("order_ids", [])[:1]:
            if oid not in self.state.verified_orders:
                res = self._lookup_order(oid)
                out.append({"name": "lookup_order", "args": {"order_id": oid}, "result": _brief(res)})
        oid = self.state.order_id
        value = (ents.get("four_digit") or ents.get("pincodes") or [None])[0]
        if oid and value and oid not in self.state.verified_orders and not self.state.locked:
            res = self._lookup_order(oid, value)
            item = {"name": "lookup_order", "args": {"order_id": oid, "verification_value": value},
                    "result": _brief(res)}
            if res.get("verified"):
                item["evidence"] = [f"orders.csv:{oid}"]
                item["result"] = {"verified": True, "note": "record is in CALL STATE"}
            out.append(item)
        return out

    def _auto_products(self, norm) -> list:
        """Look up SKUs and catalog product names the customer said, before the LLM runs."""
        found = []
        for sku in norm["entities"].get("skus", []):
            res = data.lookup_product(sku)
            if res.get("found") and len(res["products"]) == 1:
                found += res["products"]
        if not found:
            found = data.find_products_in_text(norm["text"])
        return found

    # ------------------------------------------------------------------ turn
    def _retrieval_query(self, norm_text: str) -> str:
        prev = [h["content"] for h in self.history if h["role"] == "user"][-1:]
        parts = prev + [norm_text]  # short follow-ups inherit the previous topic
        rec = self.state.verified_orders.get(self.state.order_id or "")
        if rec:  # the verified order decides which rule applies: payment method, status
            for k, v in rec.items():
                if isinstance(v, str) and len(v) < 40 and any(w in k for w in ("payment", "status", "mode")):
                    parts.append(v.replace("_", " "))
        return " ".join(parts)

    # ------------------------------------------------------------------ voice repair
    def _scripted(self, user_text, norm, reply, decision="clarify") -> dict:
        """A turn answered by code (no LLM): e.g. asking to repeat a misheard number."""
        lang = self.state.lang or reply_language(user_text)
        if self.state.turns == 0:
            reply = AI_INTRO.get(lang, AI_INTRO["en"]) + reply
        self.history += [{"role": "user", "content": user_text}, {"role": "assistant", "content": reply}]
        self.state.turns += 1
        self.last_turn = {"user": user_text, "normalized": norm["text"], "entities": norm["entities"],
                          "reply": reply, "decision": decision, "tool_calls": [], "raw_final": "[scripted]",
                          "retrieved": [], "evidence": [], "chunks": []}
        return self.last_turn

    def _repair(self, user_text, norm):
        """Misheard numbers: an order ID or verification value with the wrong number of digits."""
        last = self.history[-1]["content"] if self.history else ""
        words = re.findall(r"[^\W\d_]+", norm["text"])
        digits = "".join(re.findall(r"\d", norm["text"]))
        if not digits or len(words) > 8:
            return None
        ents = norm["entities"]
        lang = reply_language(user_text) if reply_language(user_text) != "en" else (self.state.lang or "en")
        spoken = " ".join(digits[i:i + 4] for i in range(0, len(digits), 4))
        if (not self.state.order_id and ASK_ORDER_RE.search(last) and not ents.get("order_ids")
                and 5 <= len(digits) <= 11):
            msg = {"en": f"Sorry, I heard {spoken}, which is {len(digits)} digits. Order numbers have 8 digits. "
                         "Could you say it again slowly?",
                   "hi-en": f"Sorry, maine {spoken} suna, jo {len(digits)} digits hai. Order number 8 digits ka "
                            "hota hai. Kya aap dheere se dobara bata sakte hain?",
                   "hi": f"माफ़ कीजिए, मैंने {spoken} सुना, जो {len(digits)} अंक हैं। ऑर्डर नंबर 8 अंकों का होता है। "
                         "कृपया धीरे से दोबारा बताइए।"}[lang]
            return self._scripted(user_text, norm, msg)
        if (self.state.order_id and self.state.order_id not in self.state.verified_orders
                and not self.state.locked and VERIFY_ASK_RE.search(last)
                and not (ents.get("four_digit") or ents.get("pincodes")) and 3 <= len(digits) <= 7):
            msg = {"en": f"Sorry, I heard {spoken}. Please tell me the last 4 digits of your registered phone "
                         "number, or your 6-digit delivery pincode.",
                   "hi-en": f"Sorry, maine {spoken} suna. Kripya registered phone number ke last 4 digits ya "
                            "6 digit delivery pincode batayein.",
                   "hi": f"माफ़ कीजिए, मैंने {spoken} सुना। कृपया अपने रजिस्टर्ड फ़ोन नंबर के आख़िरी 4 अंक या "
                         "6 अंकों का डिलीवरी पिनकोड बताइए।"}[lang]
            return self._scripted(user_text, norm, msg)
        return None

    def turn(self, user_text: str) -> dict:
        self._turn_retrieved = []
        norm = normalize(user_text, data.sku_prefixes())
        repaired = self._repair(user_text, norm)
        if repaired:
            return repaired
        hits = self.kb.search(self._retrieval_query(norm["text"]), k=self.top_k)
        # Approval / compensation / exception requests: always bring in who-decides & handover rules
        if any(rx.search(t) for rx in (APPROVAL_RE, DAMAGE_RE, SECURITY_RE) for t in (user_text, norm["text"])):
            have = {c.id for c, _ in hits}
            for c, sc in self.kb.search("escalation human handover who decides approval threshold "
                                        "priority resolution compensation " + norm["text"], k=4):
                if c.id not in have and c.doc_type in ("policy", "agent_guide"):
                    hits.append((c, sc))
                    have.add(c.id)
        # The precedence rule KB-00.1 decides every policy-vs-FAQ/marketing conflict: always in context
        prec = self.kb.by_id.get("KB-00.1")
        if prec and prec not in [c for c, _ in hits]:  # precedence rule always applies
            hits.append((prec, 0.0))
        self._turn_retrieved.extend(c.id for c, _ in hits)

        user_block = [f"Customer said (speech-to-text; may contain recognition errors): {user_text}"]
        if norm["text"].strip() != (user_text or "").strip():
            user_block.append(f"Normalized: {norm['text']}")
        if norm["entities"]:
            user_block.append(f"Detected: {json.dumps(norm['entities'])}")
        pre_tools = self._auto_order_tools(norm)
        for t in pre_tools:
            user_block.append(f"System already called {t['name']}({json.dumps(t['args'])}) for this turn -> "
                              f"{json.dumps(t['result'], ensure_ascii=False, default=str)[:600]}")
        auto_products = self._auto_products(norm)
        if auto_products:
            user_block.append("Catalog lookup for products the customer mentioned (products.csv): " +
                              json.dumps(auto_products, ensure_ascii=False))
        lang = reply_language(user_text)
        if lang == "en" and self.state.lang in ("hi-en", "hi") and len(user_text.split()) <= 6:
            lang = self.state.lang  # short replies like "4533" or "yes" keep the call's language
        self.state.lang = lang
        user_block.append("Reply language: " + {"en": "English", "hi-en": "Hinglish (Hindi in Roman script)",
                                                 "hi": "Hindi (Devanagari)"}[lang])
        user_block.append("\nKnowledge-base sections retrieved for this turn:\n" +
                          "\n\n".join(c.render() for c, _ in hits))

        messages = ([{"role": "system", "content": AGENT_PROMPT},
                     {"role": "system", "content": self.state.block()}]
                    + self.history
                    + [{"role": "user", "content": "\n".join(user_block)}])

        tool_log, evidence, final, raw_final = list(pre_tools), [], None, ""
        for t in pre_tools:
            evidence.extend(t.get("evidence", []))
        if auto_products:
            tool_log.append({"name": "lookup_product", "args": {"auto": True},
                             "result": {"skus": [p["sku"] for p in auto_products]}})
            evidence.extend(f"products.csv:{p['sku']}" for p in auto_products)
        date_checked = dontknow_checked = lang_checked = deadline_checked = order_checked = spec_checked = False
        when_checked = damage_checked = calc_checked = False
        for round_ in range(MAX_TOOL_ROUNDS + 1):
            force = round_ == MAX_TOOL_ROUNDS
            msg = self.llm.chat(messages, tools=TOOLS, tool_choice="none" if force else "auto",
                                max_tokens=1500)
            calls = msg.tool_calls or []
            if not calls:
                raw_final = msg.content or ""
                final = parse_final(raw_final)
                if not final["reply"]:  # empty/unusable final text: ask once more, explicitly
                    finish = getattr(self.llm, "last_finish", None)
                    messages.append({"role": "assistant", "content": raw_final})
                    messages.append({"role": "user", "content":
                                     "Write your final message now, exactly in this format:\n"
                                     "DECISION: ...\nEVIDENCE: ...\nREPLY: ..."})
                    msg = self.llm.chat(messages, tools=TOOLS, tool_choice="none", max_tokens=1500)
                    raw_final = f"[first attempt empty, finish={finish}] " + (msg.content or "")
                    final = parse_final(msg.content or "")
                bad_dates = [] if date_checked else unsupported_dates(
                    final["reply"], [user_text, norm["text"]] + [m["content"] for m in messages
                                                                  if isinstance(m.get("content"), str)])
                has_details = bool(auto_products or _dates_in(norm["text"]) or DETAILS_RE.search(norm["text"])
                                   or len(norm["text"].split()) >= 8)
                asks = ASK_ORDER_RE.search(final["reply"] or "") or final.get("decision") == "clarify"
                if (not order_checked and not self.state.order_id and has_details and asks
                        and round_ < MAX_TOOL_ROUNDS):
                    order_checked = True
                    messages.append({"role": "assistant", "content": raw_final})
                    messages.append({"role": "user", "content": (
                        "Check: do not ask a question (order ID or anything else) if this can be answered from the "
                        "documents with the details the customer already gave (product, category, payment method, "
                        "dates, order status they described): "
                        "answer it directly, using date_calc for any date. If a document says a team must handle "
                        "it, call create_ticket now (order ID is optional) and say so. If policy does not allow "
                        "the request, refuse politely. Ask for the order ID only if the answer truly depends on "
                        "this customer's order record or a detail they did not give (then ask ONE question). "
                        "If only a date is missing, state the rule instead (e.g. 'you can return it within 30 days "
                        "of delivery'). If it asks for another person's data, refuse (DECISION: refuse). If no "
                        "document covers it, say you don't have that information and offer a colleague. "
                        "Final message in the DECISION/EVIDENCE/REPLY format.")})
                    final = None
                    continue
                missing_spec = [] if spec_checked else _missing_specs(user_text + " " + norm["text"],
                                                                      auto_products, final["reply"] or "")
                if missing_spec and round_ < MAX_TOOL_ROUNDS:
                    spec_checked = True
                    messages.append({"role": "assistant", "content": raw_final})
                    messages.append({"role": "user", "content": (
                        "Check: include the exact catalog fact the customer asked about: " + "; ".join(missing_spec)
                        + ". Final message in the DECISION/EVIDENCE/REPLY format.")})
                    final = None
                    continue
                searched = any(t["name"] == "search_kb" for t in tool_log)
                if (not dontknow_checked and not searched and round_ < MAX_TOOL_ROUNDS
                        and DONT_KNOW_RE.search(final["reply"] or "")):
                    dontknow_checked = True
                    messages.append({"role": "assistant", "content": raw_final})
                    messages.append({"role": "user", "content": (
                        "Check: before saying you don't have information, call search_kb with short English "
                        "keywords for what the customer asked (consider the verified order: payment method, "
                        "status, product category). Then answer, or say you don't know if nothing covers it. "
                        "Final message in the DECISION/EVIDENCE/REPLY format.")})
                    final = None
                    continue
                if (not lang_checked and lang in ("hi-en", "hi") and final["reply"]
                        and _no_hindi(final["reply"]) and round_ < MAX_TOOL_ROUNDS):
                    lang_checked = True
                    messages.append({"role": "assistant", "content": raw_final})
                    messages.append({"role": "user", "content": (
                        f"Check: the customer speaks {'Hinglish' if lang == 'hi-en' else 'Hindi'}; rewrite the same "
                        f"reply in {'Hinglish (Roman script)' if lang == 'hi-en' else 'Hindi (Devanagari)'}. "
                        "Keep DECISION and EVIDENCE.")})
                    final = None
                    continue
                rec = self.state.verified_orders.get(self.state.order_id or "")
                if (not deadline_checked and rec and round_ < MAX_TOOL_ROUNDS
                        and final.get("decision") in (None, "answer") and ELIGIBILITY_RE.search(user_text)
                        and not (_dates_in(final["reply"]) - _dates_in(json.dumps(rec, default=str)) - {(20, "October")})):
                    deadline_checked = True
                    messages.append({"role": "assistant", "content": raw_final})
                    messages.append({"role": "user", "content": (
                        "Check: the customer needs the concrete deadline. Call date_calc with the relevant date from "
                        "the order record (e.g. delivery date) and the window from the policy section (calendar days "
                        "for return windows), then say until which date (with the year) they can do it, or that the "
                        "window has passed. Final message in the DECISION/EVIDENCE/REPLY format.")})
                    final = None
                    continue
                computed = [t["result"].get(k) for t in tool_log if t["name"] == "date_calc"
                            for k in ("after_business_days", "after_calendar_days") if t["result"].get(k)]
                if (not calc_checked and computed and round_ < MAX_TOOL_ROUNDS
                        and not any(_dates_in(c) & _dates_in(final["reply"] or "") for c in computed)):
                    calc_checked = True
                    messages.append({"role": "assistant", "content": raw_final})
                    messages.append({"role": "user", "content": (
                        f"Check: date_calc returned {' / '.join(computed)}, but your reply states a different date. "
                        "Use the date_calc result exactly (for a range like 3-5 business days, the result for the "
                        "LATEST number is the date to give). Final message in the DECISION/EVIDENCE/REPLY format.")})
                    final = None
                    continue
                user_dates = _dates_in(user_text) | _dates_in(norm["text"])
                if (not when_checked and user_dates and WHEN_RE.search(user_text) and round_ < MAX_TOOL_ROUNDS
                        and final.get("decision") in (None, "answer")
                        and not (_dates_in(final["reply"]) - user_dates - {(20, "October")})):
                    when_checked = True
                    messages.append({"role": "assistant", "content": raw_final})
                    messages.append({"role": "user", "content": (
                        "Check: the customer asked by when. Call date_calc with their start date and the timeline "
                        "from the policy (business days for refunds/logistics; for a range like 3-5 business days use "
                        "the LATEST number), then state the resulting date with the year. "
                        "Final message in the DECISION/EVIDENCE/REPLY format.")})
                    final = None
                    continue
                # 2) damage reports: is a human team required (e.g. value thresholds)?
                ticketed = any(t["name"] == "create_ticket" for t in tool_log) or self.state.tickets
                if (not damage_checked and not ticketed and round_ < MAX_TOOL_ROUNDS
                        and (DAMAGE_RE.search(user_text + " " + norm["text"])
                             or SECURITY_RE.search(user_text + " " + norm["text"]))
                        and final.get("decision") in (None, "answer", "clarify")):
                    damage_checked = True
                    price = ", ".join(f"{p.get('sku')}: " + ", ".join(f"{k}={v}" for k, v in p.items() if "price" in k)
                                      for p in auto_products) or "unknown"
                    messages.append({"role": "assistant", "content": raw_final})
                    messages.append({"role": "user", "content": (
                        f"Check: this is a damage/wrong/missing-item or account-security report (catalog price: "
                        f"{price}). Re-read the damaged-item, payment/security and escalation sections: if they say a "
                        "specific team must handle this case "
                        "(e.g. above a value threshold), call create_ticket now and tell the customer it has been "
                        "passed to that team (DECISION: escalate). Otherwise keep your answer. "
                        "Final message in the DECISION/EVIDENCE/REPLY format.")})
                    final = None
                    continue
                if bad_dates and round_ < MAX_TOOL_ROUNDS:
                    date_checked = True
                    messages.append({"role": "assistant", "content": raw_final})
                    messages.append({"role": "user", "content": (
                        f"Check: your reply states {', '.join(bad_dates)}, which no tool result or document gives. "
                        "Call date_calc to compute it (business days for refund/logistics timelines, calendar days "
                        "for return windows; for a range like 5-7 business days give the LATEST date). "
                        "Then write the corrected final message in the DECISION/EVIDENCE/REPLY format.")})
                    final = None
                    continue
                break
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [_tool_call_dict(c) for c in calls]})
            for c in calls:
                try:
                    args = json.loads(c.function.arguments or "{}") or {}
                except json.JSONDecodeError:
                    args = {}
                if c.function.name == "respond":  # tolerate models that still try a "respond" tool
                    final = {"reply": args.get("reply", ""), "decision": args.get("decision"),
                             "evidence": args.get("evidence") or []}
                    result = {"ok": True}
                else:
                    result = self._call_tool(c.function.name, args)
                    tool_log.append({"name": c.function.name, "args": args,
                                     "result": _brief(result)})
                    evidence.extend(result.get("evidence", []) if isinstance(result, dict) else [])
                messages.append({"role": "tool", "tool_call_id": c.id,
                                 "content": json.dumps(result, ensure_ascii=False, default=str)[:6000]})
            if final is not None:
                break

        final = final or {"reply": "", "decision": None}
        reply = _clean_reply(final.get("reply") or "")
        tickets_this_turn = [t for t in tool_log if t["name"] == "create_ticket" and t["result"].get("created")]
        verified_now = any(t["name"] == "lookup_order" and t["result"].get("verified") is True
                           and t["args"].get("verification_value") for t in tool_log)
        decision = final.get("decision")
        if decision not in ("answer", "clarify", "escalate", "refuse"):
            decision = "escalate" if tickets_this_turn else ("clarify" if reply.endswith("?") else "answer")
        if decision == "clarify" and not self.state.order_id and _answers_first(reply):
            decision = "answer"  # it gave the answer and only added a follow-up question
        if verified_now and decision == "clarify":
            decision = "answer"  # verification completed this turn: that is a resolved step, not a question

        # Guard: an escalation must be backed by a real ticket (never claim what code didn't do).
        if decision == "escalate" and not tickets_this_turn and not self.state.tickets:
            summary = (f"Customer request: {user_text.strip()[:300]} | Agent reply: {reply[:300]} | "
                       f"Order: {self.state.order_id or 'n/a'}; verified: {bool(self.state.verified_orders)}")
            res = self._create_ticket(summary=summary, reason="Escalated by agent decision", intent="handover")
            tool_log.append({"name": "create_ticket", "args": {"summary": summary, "auto": True}, "result": _brief(res)})

        if not reply:
            reply = {"hi": "माफ़ कीजिए, कृपया दोबारा बताइए।", "hi-en": "Sorry, kya aap dobara bata sakte hain?"}.get(
                lang, "Sorry, could you say that again?")
        if self.state.turns == 0 and not re.search(r"\bAI\b|एआई", reply):
            reply = re.sub(r"^(?:namaste|namaskar|hi|hello|hey|नमस्ते)[!,.]*\s*", "", reply, flags=re.I)
            reply = AI_INTRO.get(lang, AI_INTRO["en"]) + reply

        cited = [e for e in (final.get("evidence") or []) if isinstance(e, str)]
        known = set(self.kb.by_id)
        evidence = list(dict.fromkeys([e.strip() for e in cited if e.strip() in known or ".csv:" in e] + evidence))
        retrieved = list(dict.fromkeys(self._turn_retrieved))

        self.history += [{"role": "user", "content": user_text},
                         {"role": "assistant", "content": reply}]
        self.state.turns += 1
        self.last_turn = {
            "user": user_text, "normalized": norm["text"], "entities": norm["entities"],
            "reply": reply, "decision": decision, "tool_calls": tool_log, "raw_final": raw_final,
            "retrieved": retrieved, "evidence": evidence,
            "chunks": [{"id": cid, "text": self.kb.by_id[cid].text} for cid in
                       (([e for e in evidence if e in known]) or retrieved) if cid in self.kb.by_id],
        }
        return self.last_turn

    def respond(self, user_text: str) -> str:
        """Brief-compatible interface: text in, reply text out."""
        return self.turn(user_text)["reply"]


HINDI_MONTHS = {"जनवरी": "January", "फ़रवरी": "February", "फरवरी": "February", "मार्च": "March",
                "अप्रैल": "April", "मई": "May", "जून": "June", "जुलाई": "July", "अगस्त": "August",
                "सितंबर": "September", "सितम्बर": "September", "अक्टूबर": "October", "अक्तूबर": "October",
                "नवंबर": "November", "नवम्बर": "November", "दिसंबर": "December", "दिसम्बर": "December"}
_MONTH_LIST = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
               "October", "November", "December"]


DETAILS_RE = re.compile(r"upi|card|net ?banking|wallet|\bcod\b|cash|refund|return|damag|torn|broken|crack|"
                        r"gst|invoice|bulk|ship|discount|price|offer|coupon|pickup|शिप|भेज|रद्द|कैंसल|इनवॉइस|"
                        r"warranty|assured|deliver|cancel|fee|charge|compensat|phone number|रिफंड|रिटर्न|लौटा|"
                        r"वापस|टूट|डिलीवरी|कैंसिल", re.I)


VERIFY_ASK_RE = re.compile(r"last 4|last four|pincode|pin code|aakhri|आख़िरी|आखिरी|पिनकोड", re.I)


ASK_ORDER_RE = re.compile(r"order (?:id|number|no)|ऑर्डर (?:आईडी|आई डी|नंबर|संख्या)|order ki id", re.I)


ELIGIBILITY_RE = re.compile(r"\b(return|returns|returnable|replace|replacement|exchange|warranty|claim)\b|"
                            r"lauta|lautana|wapas kar|रिटर्न|लौटा|वापस", re.I)


SECURITY_RE = re.compile(r"never placed|didn'?t place|did not place|not placed by me|unauthori[sz]ed|fraud|hack|"
                         r"someone (?:else )?(?:used|ordered|placed)|not me|without my|stolen|scam|"
                         r"maine (?:order )?nahi kiya|मैंने .*नहीं किया|धोखा", re.I)


WHEN_RE = re.compile(r"\bwhen\b|by when|how long|kab|kitne din|कब|कितने दिन|tak aa", re.I)


DAMAGE_RE = re.compile(r"damag|broken|crack|torn|dent|defect|faulty|leak|smash|shatter|missing|wrong item|"
                       r"tuta|toota|kharab|phata|टूट|खराब|फटा|गलत", re.I)
WARRANTY_RE = re.compile(r"warrant|guarantee|assured|वारंटी|गारंटी", re.I)


def _missing_specs(text: str, products: list, reply: str) -> list:
    """Catalog facts the customer asked about that the reply leaves out (e.g. warranty months)."""
    out = []
    if not products or not WARRANTY_RE.search(text or ""):
        return out
    for p in products[:1]:
        for k, v in p.items():
            if "warranty" in k.lower() and str(v).strip().isdigit() and int(v) > 0:
                if not re.search(rf"(?<!\d){v}(?!\d)", reply):
                    out.append(f"{p.get('sku', 'product')} has {k} = {v} (say '{v} months')")
    return out


APPROVAL_RE = re.compile(r"approv|compensat|manager|supervisor|senior|exception|refund (?:it )?now|"
                         r"right now|abhi (?:kar|de|approve)|kar do na|de do|jaldi|escalat|complain|"
                         r"मंज़ूर|मंजूर|मुआवज", re.I)


DONT_KNOW_RE = re.compile(r"don'?t have (?:any )?information|do not have (?:any )?information|"
                          r"jaankari nahi|jankari nahi|जानकारी नहीं|pata nahi", re.I)


def _dates_in(text: str) -> set:
    """(day, month) pairs mentioned in text: '27 October', 'October 27', '27 अक्टूबर', '2026-10-27'."""
    out = set()
    t = text or ""
    for hi, en in HINDI_MONTHS.items():
        t = t.replace(hi, en)
    months = "|".join(_MONTH_LIST)
    for d, m in re.findall(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({months})", t, re.I):
        out.add((int(d), m.capitalize()))
    for m, d in re.findall(rf"\b({months})\s+(\d{{1,2}})\b", t, re.I):
        out.add((int(d), m.capitalize()))
    for y, mo, d in re.findall(r"\b(20\d\d)-(\d{2})-(\d{2})\b", t):
        if 1 <= int(mo) <= 12:
            out.add((int(d), _MONTH_LIST[int(mo) - 1]))
    return out


def unsupported_dates(reply: str, sources: list) -> list:
    """Dates in the reply that appear in no source (customer text, documents, tool results)."""
    allowed = {(20, "October")}  # today
    for src in sources:
        allowed |= _dates_in(src if isinstance(src, str) else json.dumps(src, ensure_ascii=False))
    return [f"{d} {m}" for d, m in sorted(_dates_in(reply) - allowed)]


def parse_final(text: str) -> dict:
    """Parse 'DECISION: ... / EVIDENCE: ... / REPLY: ...' (lenient; falls back to the whole text)."""
    text = (text or "").strip()
    dec = re.search(r"DECISION\s*[:=]\s*\W*(answer|clarify|escalate|refuse)", text, re.I)
    ev = re.search(r"EVIDENCE\s*[:=]\s*(.*?)(?:\n|REPLY\s*[:=]|$)", text, re.I | re.S)
    rep = re.search(r"REPLY\s*[:=]\s*(.*)$", text, re.I | re.S)
    if rep:
        reply = rep.group(1)
    else:  # no REPLY label: strip any label lines / salvaged tool-call debris
        reply = re.sub(r"(?im)^\s*(DECISION|EVIDENCE)\s*[:=].*$", "", text)
        reply = re.sub(r'^\{?\s*"name"\s*:\s*"\w+"\s*,\s*"arguments"\s*:\s*', "", reply).rstrip('"} ')
        reply = re.split(r"\s(?:decision|evidence)\s*:", reply, flags=re.I)[0]
    evidence = []
    if ev and ev.group(1).strip().lower() not in ("none", "-", "[]", ""):
        evidence = re.findall(r"[\w.]+\.csv:[\w-]+|\b[A-Z][A-Z0-9]{1,5}-\d+(?:\.\d+)?\b", ev.group(1))
    return {"reply": reply.strip().strip('"'), "decision": dec.group(1).lower() if dec else None,
            "evidence": evidence}


def _tool_call_dict(c) -> dict:
    """Assistant tool call to send back. Keeps Gemini's thought signature (extra_content), which
    Gemini 3 requires on the next request; other providers never send it."""
    d = {"id": c.id, "type": "function",
         "function": {"name": c.function.name, "arguments": c.function.arguments or "{}"}}
    extra = getattr(c, "extra_content", None)
    if extra is None:
        extra = (getattr(c, "model_extra", None) or {}).get("extra_content")
    if extra:
        d["extra_content"] = extra
    return d


def _brief(result):
    """Shorter tool result for logs."""
    if not isinstance(result, dict):
        return result
    out = {k: v for k, v in result.items() if k not in ("items", "sections")}
    if "items" in result:
        out["items"] = len(result["items"])
    if "sections" in result:
        out["sections"] = len(result["sections"])
    return out


MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"


NUM_WORDS = {"two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
             "nine": "9", "ten": "10", "twelve": "12", "fifteen": "15", "thirty": "30", "forty-eight": "48"}


def _answers_first(reply: str) -> bool:
    """True when the reply's first real sentence states a fact with a number (not a question)."""
    body = re.sub(r"^.*?\b(?:assistant|assistant hoon|सहायक हूँ)[.!।]?\s*", "", reply, count=1, flags=re.I)
    first = re.split(r"(?<=[.!?।])\s+", body.strip())[0] if body.strip() else ""
    return bool(first) and not first.rstrip().endswith("?") and bool(re.search(r"\d", first))


def _clean_reply(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("\u202f", " ")
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    # spoken dates need the year: "27 October" -> "27 October 2026" (dataset is frozen in 2026)
    text = re.sub(rf"\b(\d{{1,2}}(?:st|nd|rd|th)? (?:{MONTHS}))\b(?!,? \d{{4}})", r"\1 2026", text)
    text = re.sub(rf"\b((?:{MONTHS}) \d{{1,2}}(?:st|nd|rd|th)?)\b(?!,? \d{{4}})", r"\1, 2026", text)
    text = re.sub(r"\b(" + "|".join(NUM_WORDS) + r")\b(?=[\s-]+[a-z])",
                  lambda m: NUM_WORDS[m.group(1).lower()], text, flags=re.I)
    text = re.sub(r"[*_#`>]+", "", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)
    text = re.sub(r"\s*\n+\s*", " ", text)
    return text.strip()
