"""
Legal research tools for external API integration.
Follows SOLID principles with modular, testable tool implementations.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
from functools import lru_cache

from app.models import LegalSource, SourceType, ToolCallResult
from app.config.settings import COHERE_API_KEY

logger = logging.getLogger("legal_research_tools")


class LegalResearchTool(ABC):
    """Abstract base class for legal research tools following SOLID principles"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the research tool"""
        pass

    @abstractmethod
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get JSON schema for tool parameters"""
        pass

    async def call(self, **kwargs) -> ToolCallResult:
        """Call the tool and return a standardized result"""
        start_time = datetime.now()

        try:
            result = await self.execute(**kwargs)
            sources = self._extract_sources_from_result(result)

            return ToolCallResult(
                tool_name=self.name,
                arguments=kwargs,
                result=result,
                success=True,
                timestamp=datetime.now(),
                sources_generated=sources
            )

        except Exception as e:
            logger.error(f"Tool {self.name} failed: {str(e)}")
            return ToolCallResult(
                tool_name=self.name,
                arguments=kwargs,
                result={"error": str(e)},
                success=False,
                timestamp=datetime.now(),
                sources_generated=[]
            )

    @abstractmethod
    def _extract_sources_from_result(self, result: Dict[str, Any]) -> List[LegalSource]:
        """Extract legal sources from tool results"""
        pass


class NigerianStatuteSearchTool(LegalResearchTool):
    """Tool for searching Nigerian statutes and legal codes"""

    def __init__(self):
        super().__init__(
            name="search_nigerian_statutes",
            description="Search Nigerian statutes, codes, and legal provisions"
        )

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Legal search query for Nigerian statutes (e.g., 'insider trading definition')"
                },
                "jurisdiction": {
                    "type": "string",
                    "description": "Specific Nigerian jurisdiction (e.g., 'Federal', 'Lagos', 'Abuja')",
                    "default": "Nigeria"
                }
            },
            "required": ["query"]
        }

    async def execute(self, query: str, jurisdiction: str = "Nigeria") -> Dict[str, Any]:
        """Search Nigerian legal databases for relevant statutes"""
        logger.info(f"Searching Nigerian statutes: {query} in {jurisdiction}")

        # Mock implementation - replace with real Nigerian legal database API
        await asyncio.sleep(0.5)  # Simulate API call

        mock_statutes = [
            {
                "title": "Investment and Securities Act, 2007",
                "summary": "Federal legislation governing securities and investments in Nigeria",
                "relevance": 0.95,
                "citation": "Investment and Securities Act, 2007, Cap. I24, LFN 2004",
                "year": 2007,
                "sections": ["Section 112", "Section 113", "Section 114"],
                "tool_used": self.name
            },
            {
                "title": "Companies and Allied Matters Act, 2020",
                "summary": "Governs company formation and regulation in Nigeria",
                "relevance": 0.78,
                "citation": "Companies and Allied Matters Act, 2020, Cap. C20, LFN 2020",
                "year": 2020,
                "sections": ["Section 330", "Section 331"],
                "tool_used": self.name
            },
            {
                "title": "Nigerian Criminal Code Act",
                "summary": "Federal criminal legislation applicable nationwide",
                "relevance": 0.65,
                "citation": "Criminal Code Act, Cap. C38, LFN 2004",
                "year": 2004,
                "sections": ["Chapter 12", "Chapter 13"],
                "tool_used": self.name
            }
        ]

        # Filter statutes based on query relevance
        relevant_statutes = self._filter_by_query_relevance(query, mock_statutes)

        return {
            "success": True,
            "statutes": relevant_statutes,
            "query": query,
            "jurisdiction": jurisdiction,
            "total_found": len(relevant_statutes),
            "timestamp": datetime.now().isoformat()
        }

    def _filter_by_query_relevance(self, query: str, statutes: List[Dict]) -> List[Dict]:
        """Filter statutes based on query relevance (simple keyword matching)"""
        query_lower = query.lower()
        keywords = ["securities", "investment", "insider", "trading", "fraud", "criminal", "company"]

        relevant_statutes = []
        for statute in statutes:
            title_lower = statute["title"].lower()
            summary_lower = statute["summary"].lower()

            # Increase relevance for matching keywords
            relevance_boost = 0
            for keyword in keywords:
                if keyword in query_lower and keyword in title_lower:
                    relevance_boost += 0.1
                elif keyword in query_lower and keyword in summary_lower:
                    relevance_boost += 0.05

            statute["relevance"] = min(1.0, statute["relevance"] + relevance_boost)
            if statute["relevance"] > 0.5:  # Only include relevant statutes
                relevant_statutes.append(statute)

        return sorted(relevant_statutes, key=lambda x: x["relevance"], reverse=True)

    def _extract_sources_from_result(self, result: Dict[str, Any]) -> List[LegalSource]:
        """Extract legal sources from statute search results"""
        sources = []

        for statute in result.get("statutes", []):
            source = LegalSource(
                title=statute["title"],
                content_preview=statute["summary"][:200],
                relevance_score=statute["relevance"],
                source_type=SourceType.STATUTE,
                citation=statute["citation"],
                jurisdiction="Nigeria",
                year=statute["year"],
                metadata={
                    "tool_used": self.name,
                    "sections": statute.get("sections", []),
                    "search_query": result.get("query")
                }
            )
            sources.append(source)

        return sources


class CaseLawSearchTool(LegalResearchTool):
    """Tool for searching Nigerian case law and precedents"""

    def __init__(self):
        super().__init__(
            name="search_case_precedents",
            description="Search Nigerian case law and court precedents"
        )

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "legal_issue": {
                    "type": "string",
                    "description": "Legal issue to research (e.g., 'insider trading enforcement')"
                },
                "jurisdiction": {
                    "type": "string",
                    "description": "Court jurisdiction (e.g., 'Federal High Court', 'Supreme Court')",
                    "default": "Nigeria"
                },
                "year_range": {
                    "type": "string",
                    "description": "Year range for cases (e.g., '2010-2024')",
                    "default": None
                }
            },
            "required": ["legal_issue"]
        }

    async def execute(self, legal_issue: str, jurisdiction: str = "Nigeria", year_range: str = None) -> Dict[str, Any]:
        """Search Nigerian case law databases for relevant precedents"""
        logger.info(f"Searching case law: {legal_issue} in {jurisdiction}")

        # Mock implementation - replace with real Nigerian case law API
        await asyncio.sleep(0.7)  # Simulate API call

        mock_cases = [
            {
                "case_name": "Securities and Exchange Commission v. Alhaji Ibrahim (2021)",
                "court": "Federal High Court, Lagos",
                "citation": "SEC v. Alhaji (2021) FHC/L/CS/1254/2020",
                "ruling": "The court held that insider trading requires proof of both possession of non-public information and intent to use it for trading advantage",
                "relevance_score": 0.92,
                "year": 2021,
                "legal_principle": "Insider trading elements",
                "tool_used": self.name
            },
            {
                "case_name": "Central Bank of Nigeria v. Sterling Bank Plc (2022)",
                "court": "Supreme Court of Nigeria",
                "citation": "CBN v. Sterling Bank (2022) 13 NWLR (Pt. 1593) 123",
                "ruling": "The Supreme Court clarified the standard of proof required for financial misconduct cases",
                "relevance_score": 0.78,
                "year": 2022,
                "legal_principle": "Standard of proof in financial cases",
                "tool_used": self.name
            },
            {
                "case_name": "FRCN v. Skye Bank Plc (2020)",
                "court": "Court of Appeal, Lagos",
                "citation": "FRCN v. Skye Bank (2020) 17 NWLR (Pt. 1529) 456",
                "ruling": "The Court of Appeal addressed issues of regulatory enforcement and due process",
                "relevance_score": 0.71,
                "year": 2020,
                "legal_principle": "Regulatory enforcement procedures",
                "tool_used": self.name
            }
        ]

        # Filter cases based on legal issue relevance
        relevant_cases = self._filter_by_legal_issue(legal_issue, mock_cases)

        return {
            "success": True,
            "cases": relevant_cases,
            "legal_issue": legal_issue,
            "jurisdiction": jurisdiction,
            "total_found": len(relevant_cases),
            "timestamp": datetime.now().isoformat()
        }

    def _filter_by_legal_issue(self, legal_issue: str, cases: List[Dict]) -> List[Dict]:
        """Filter cases based on legal issue relevance"""
        issue_lower = legal_issue.lower()

        relevant_cases = []
        for case in cases:
            ruling_lower = case["ruling"].lower()
            principle_lower = case["legal_principle"].lower()

            # Simple relevance calculation based on keyword overlap
            relevance_boost = 0
            issue_words = issue_lower.split()

            for word in issue_words:
                if len(word) > 3:  # Skip short words
                    if word in ruling_lower:
                        relevance_boost += 0.05
                    if word in principle_lower:
                        relevance_boost += 0.08

            case["relevance_score"] = min(1.0, case["relevance_score"] + relevance_boost)
            if case["relevance_score"] > 0.6:
                relevant_cases.append(case)

        return sorted(relevant_cases, key=lambda x: x["relevance_score"], reverse=True)

    def _extract_sources_from_result(self, result: Dict[str, Any]) -> List[LegalSource]:
        """Extract legal sources from case law search results"""
        sources = []

        for case in result.get("cases", []):
            source = LegalSource(
                title=case["case_name"],
                content_preview=case["ruling"][:200],
                relevance_score=case["relevance_score"],
                source_type=SourceType.CASE,
                citation=case["citation"],
                jurisdiction="Nigeria",
                year=case["year"],
                court=case["court"],
                metadata={
                    "tool_used": self.name,
                    "legal_principle": case.get("legal_principle"),
                    "search_issue": result.get("legal_issue")
                }
            )
            sources.append(source)

        return sources


class RegulationSearchTool(LegalResearchTool):
    """Tool for searching industry regulations and compliance requirements"""

    def __init__(self):
        super().__init__(
            name="search_regulations",
            description="Search industry-specific regulations and compliance requirements"
        )

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "industry": {
                    "type": "string",
                    "description": "Industry sector (e.g., 'banking', 'telecommunications', 'oil and gas')"
                },
                "regulation_type": {
                    "type": "string",
                    "description": "Type of regulation (e.g., 'compliance', 'licensing', 'reporting')",
                    "default": None
                },
                "jurisdiction": {
                    "type": "string",
                    "description": "Regulatory jurisdiction",
                    "default": "Nigeria"
                }
            },
            "required": ["industry"]
        }

    async def execute(self, industry: str, regulation_type: str = None, jurisdiction: str = "Nigeria") -> Dict[str, Any]:
        """Search regulatory databases for industry-specific requirements"""
        logger.info(f"Searching regulations for {industry} industry in {jurisdiction}")

        # Mock implementation - replace with real regulatory API
        await asyncio.sleep(0.6)  # Simulate API call

        regulation_database = {
            "banking": [
                {
                    "title": "Central Bank of Nigeria Prudential Guidelines",
                    "summary": "Comprehensive prudential guidelines for Nigerian banks",
                    "relevance": 0.88,
                    "regulator": "Central Bank of Nigeria",
                    "year": 2023,
                    "sections": ["Risk Management", "Capital Adequacy", "Corporate Governance"]
                },
                {
                    "title": "Banking and Other Financial Institutions Act",
                    "summary": "Primary legislation governing banking operations in Nigeria",
                    "relevance": 0.91,
                    "regulator": "National Assembly",
                    "year": 2020,
                    "sections": ["Licensing Requirements", "Operational Standards"]
                }
            ],
            "telecommunications": [
                {
                    "title": "Nigerian Communications Act",
                    "summary": "Legal framework for telecommunications in Nigeria",
                    "relevance": 0.85,
                    "regulator": "Nigerian Communications Commission",
                    "year": 2003,
                    "sections": ["Licensing", "Service Quality", "Consumer Protection"]
                }
            ],
            "oil and gas": [
                {
                    "title": "Petroleum Industry Act",
                    "summary": "Comprehensive legislation for oil and gas sector",
                    "relevance": 0.90,
                    "regulator": "Nigerian Upstream Petroleum Regulatory Commission",
                    "year": 2021,
                    "sections": ["Upstream Operations", "Downstream Operations", "Environmental Standards"]
                }
            ]
        }

        industry_lower = industry.lower()
        regulations = regulation_database.get(industry_lower, [
            {
                "title": f"General {industry.title()} Regulations",
                "summary": f"Regulatory framework for {industry} sector in Nigeria",
                "relevance": 0.70,
                "regulator": "Regulatory Authority",
                "year": 2022,
                "sections": ["Compliance Requirements"]
            }
        ])

        return {
            "success": True,
            "regulations": regulations,
            "industry": industry,
            "regulation_type": regulation_type,
            "jurisdiction": jurisdiction,
            "total_found": len(regulations),
            "timestamp": datetime.now().isoformat()
        }

    def _extract_sources_from_result(self, result: Dict[str, Any]) -> List[LegalSource]:
        """Extract legal sources from regulation search results"""
        sources = []

        for regulation in result.get("regulations", []):
            source = LegalSource(
                title=regulation["title"],
                content_preview=regulation["summary"][:200],
                relevance_score=regulation["relevance"],
                source_type=SourceType.REGULATION,
                jurisdiction="Nigeria",
                year=regulation["year"],
                metadata={
                    "tool_used": self.name,
                    "regulator": regulation.get("regulator"),
                    "sections": regulation.get("sections", []),
                    "industry": result.get("industry")
                }
            )
            sources.append(source)

        return sources


class LegalResearchToolManager:
    """Manager for legal research tools following single responsibility principle"""

    def __init__(self):
        self.tools = {
            "search_nigerian_statutes": NigerianStatuteSearchTool(),
            "search_case_precedents": CaseLawSearchTool(),
            "search_regulations": RegulationSearchTool()
        }

    def get_tool(self, tool_name: str) -> Optional[LegalResearchTool]:
        """Get a specific tool by name"""
        return self.tools.get(tool_name)

    def get_all_tools(self) -> Dict[str, LegalResearchTool]:
        """Get all available tools"""
        return self.tools

    def get_tools_for_query(self, query: str) -> List[LegalResearchTool]:
        """Get relevant tools based on query analysis"""
        query_lower = query.lower()
        relevant_tools = []

        # Simple heuristic for tool selection
        if any(keyword in query_lower for keyword in ["statute", "law", "act", "code", "section"]):
            relevant_tools.append(self.tools["search_nigerian_statutes"])

        if any(keyword in query_lower for keyword in ["case", "precedent", "court", "ruling", "judgment"]):
            relevant_tools.append(self.tools["search_case_precedents"])

        if any(keyword in query_lower for keyword in ["regulation", "compliance", "industry", "sector"]):
            relevant_tools.append(self.tools["search_regulations"])

        # Default to all tools if no specific keywords found
        if not relevant_tools:
            relevant_tools = list(self.tools.values())

        return relevant_tools


# Singleton instance for dependency injection
@lru_cache(maxsize=1)
def get_legal_research_manager() -> LegalResearchToolManager:
    """Get singleton instance of legal research tool manager"""
    return LegalResearchToolManager()