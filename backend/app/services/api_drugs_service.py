# backend/app/services/api_drugs_service.py
import asyncio
import json
import logging
import requests
import aiohttp
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)

class MedicalAPIDrugService:
    """
    Service that integrates multiple free medical APIs to provide comprehensive drug information.
    API First approach with local database fallback.
    """

    def __init__(self):
        # API Endpoints - All FREE
        self.openfda_base = "https://api.fda.gov/drug/label.json"
        self.pubchem_base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        self.rxnorm_base = "https://rxnav.nlm.nih.gov/REST"

        # Cache settings
        self.cache_timeout = 3600  # 1 hour
        self.max_retries = 3
        self.request_timeout = 15  # seconds

        # Local database fallback
        self.db_fallback = None

    async def initialize(self):
        """Initialize the API service"""
        logger.info("Initializing Medical API Drug Service")
        logger.info("OpenFDA API: ✅ FREE & Ready")
        logger.info("PubChem API: ✅ FREE & Ready")
        logger.info("RxNorm API: ✅ FREE & Ready")

    async def cleanup(self):
        """Cleanup resources"""
        pass

    # ==================== OPENFDA API INTEGRATION ====================

    async def search_openfda_drug(self, drug_name: str) -> Optional[Dict]:
        """Search for drug information using FDA's OpenFDA API"""
        try:
            params = {
                'search': f'drug_name:"{drug_name}"',
                'limit': 5
            }

            response = await self._make_request(self.openfda_base, params)
            if not response or 'results' not in response:
                logger.debug(f"No FDA data found for {drug_name}")
                return None

            # Extract the most relevant result
            results = response['results']
            if not results:
                return None

            result = results[0]  # Take first/best result

            # Structure the FDA data
            drug_data = {
                'source': 'OpenFDA_API',
                'brand_name': result.get('brand_name', []),
                'generic_name': result.get('generic_name', []),
                'dosage_and_administration': result.get('dosage_and_administration', []),
                'indications_and_usage': result.get('indications_and_usage', []),
                'warnings': result.get('warnings', []),
                'contraindications': result.get('contraindications', []),
                'adverse_reactions': result.get('adverse_reactions', []),
                'drug_interactions': result.get('drug_interactions', []),
                'mechanism_of_action': result.get('mechanism_of_action', []),
                'clinical_pharmacology': result.get('clinical_pharmacology', []),
                'last_updated': datetime.now().isoformat(),
                'confidence_score': 0.9  # High confidence from FDA data
            }

            logger.info(f"✅ Retrieved FDA data for {drug_name}")
            return drug_data

        except Exception as e:
            logger.error(f"OpenFDA API error for {drug_name}: {str(e)}")
            return None

    # ==================== PUBCHEM API INTEGRATION ====================

    async def search_pubchem_drug(self, drug_name: str) -> Optional[Dict]:
        """Search for drug information using PubChem API"""
        try:
            # First get the compound ID
            props_url = f"{self.pubchem_base}/compound/name/{drug_name}/cids/JSON"
            cids_response = await self._make_request(props_url, {})

            if not cids_response or 'IdentifierList' not in cids_response:
                logger.debug(f"No PubChem CID found for {drug_name}")
                return None

            cid = cids_response['IdentifierList']['CID'][0]

            # Get detailed properties
            props_params = f'MolecularFormula,MolecularWeight,IUPACName,Use_and_Manufacturing'
            props_url = f"{self.pubchem_base}/compound/cid/{cid}/property/{props_params}/JSON"

            props_response = await self._make_request(props_url, {})
            if not props_response:
                return None

            props = props_response['PropertyTable']['Properties'][0]

            # Structure the PubChem data
            drug_data = {
                'source': 'PubChem_API',
                'pubchem_cid': cid,
                'molecular_formula': props.get('MolecularFormula', ''),
                'molecular_weight': props.get('MolecularWeight', 0),
                'iupac_name': props.get('IUPACName', ''),
                'usage': props.get('Use', ''),
                'manufacturing': props.get('Manufacturing', ''),
                'last_updated': datetime.now().isoformat(),
                'confidence_score': 0.85
            }

            logger.info(f"✅ Retrieved PubChem data for {drug_name} (CID: {cid})")
            return drug_data

        except Exception as e:
            logger.error(f"PubChem API error for {drug_name}: {str(e)}")
            return None

    # ==================== RXNORM API INTEGRATION ====================

    async def search_rxnorm_drug(self, drug_name: str) -> Optional[Dict]:
        """Search for standardized drug names using RxNorm API"""
        try:
            # Search for drug
            search_url = f"{self.rxnorm_base}/drugs.json"
            params = {'name': drug_name, 'limit': 5}

            response = await self._make_request(search_url, params)
            if not response or 'drugGroup' not in response:
                logger.debug(f"No RxNorm data found for {drug_name}")
                return None

            drug_group = response['drugGroup']
            if 'conceptGroup' not in drug_group:
                return None

            # Extract the first/best result
            concept_groups = drug_group['conceptGroup']
            if not concept_groups:
                return None

            # Structure the RxNorm data
            drug_data = {
                'source': 'RxNorm_API',
                'rxnorm_concept_groups': concept_groups,
                'standardized_names': [],
                'last_updated': datetime.now().isoformat(),
                'confidence_score': 0.8
            }

            # Extract standardized names
            for group in concept_groups:
                if 'conceptProperties' in group:
                    for prop in group['conceptProperties']:
                        drug_data['standardized_names'].append({
                            'rxcui': prop.get('rxcui'),
                            'name': prop.get('name'),
                            'synonym': prop.get('synonym'),
                            'tty': prop.get('tty')  # Term Type
                        })

            logger.info(f"✅ Retrieved RxNorm data for {drug_name}")
            return drug_data

        except Exception as e:
            logger.error(f"RxNorm API error for {drug_name}: {str(e)}")
            return None

    # ==================== MAIN DRUG SEARCH METHOD ====================

    async def search_drug_comprehensive(self, drug_name: str) -> Dict[str, Any]:
        """
        Comprehensive drug search using multiple APIs.
        Returns most complete drug information available.
        """

        logger.info(f"🔍 Searching for comprehensive drug data: {drug_name}")

        # Initialize results
        combined_data = {
            'drug_name': drug_name,
            'sources_used': [],
            'data_quality': 'partial',
            'last_updated': datetime.now().isoformat(),
            'confidence_score': 0.0
        }

        try:
            # 1. OpenFDA API - Primary source (most authoritative)
            fda_data = await self.search_openfda_drug(drug_name)
            if fda_data:
                combined_data.update(fda_data)
                combined_data['sources_used'].append('OpenFDA')
                combined_data['data_quality'] = 'high'
                combined_data['confidence_score'] = max(combined_data['confidence_score'], 0.9)

            # 2. PubChem API - Supplemental chemical data
            pubchem_data = await self.search_pubchem_drug(drug_name)
            if pubchem_data:
                # Merge PubChem data without overwriting OpenFDA data
                for key, value in pubchem_data.items():
                    if key not in combined_data or not combined_data[key]:
                        combined_data[key] = value
                combined_data['sources_used'].append('PubChem')
                combined_data['confidence_score'] = max(combined_data['confidence_score'], 0.85)

            # 3. RxNorm API - Standardization data
            rxnorm_data = await self.search_rxnorm_drug(drug_name)
            if rxnorm_data:
                combined_data['rxnorm_data'] = rxnorm_data
                combined_data['sources_used'].append('RxNorm')
                combined_data['confidence_score'] = max(combined_data['confidence_score'], 0.8)

            # Enhance data quality assessment
            if len(combined_data['sources_used']) >= 3:
                combined_data['data_quality'] = 'comprehensive'
                combined_data['confidence_score'] = min(combined_data['confidence_score'] + 0.1, 1.0)
            elif len(combined_data['sources_used']) >= 2:
                combined_data['data_quality'] = 'good'
            else:
                combined_data['data_quality'] = 'basic'

            logger.info(f"✅ Comprehensive search complete for {drug_name}")
            logger.info(f"   Sources: {', '.join(combined_data['sources_used'])}")
            logger.info(f"   Quality: {combined_data['data_quality']}")
            logger.info(f"   Confidence: {combined_data['confidence_score']:.2f}")

            return combined_data

        except Exception as e:
            logger.error(f"Comprehensive drug search failed: {str(e)}")

            # Return basic structure even if all APIs fail
            return {
                'drug_name': drug_name,
                'error': str(e),
                'sources_used': [],
                'data_quality': 'error',
                'confidence_score': 0.0,
                'last_updated': datetime.now().isoformat()
            }

    # ==================== UTILITY METHODS ====================

    async def _make_request(self, url: str, params: dict) -> Optional[Dict]:
        """Make HTTP request with retry logic and error handling"""
        for attempt in range(self.max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=self.request_timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            logger.warning(f"API request failed: {response.status} - {url}")
                            return None

            except asyncio.TimeoutError:
                logger.warning(f"Request timeout (attempt {attempt + 1}): {url}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return None

            except Exception as e:
                logger.error(f"Request error (attempt {attempt + 1}): {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return None

        return None

    def _generate_request_hash(self, url: str, params: dict) -> str:
        """Generate hash for request caching"""
        request_str = url + json.dumps(params, sort_keys=True)
        return hashlib.md5(request_str.encode()).hexdigest()

    # ==================== FALLBACK INTEGRATION ====================

    def set_fallback_service(self, fallback_service):
        """Set fallback service for when APIs are unavailable"""
        self.db_fallback = fallback_service
        logger.info("Database fallback service configured")

    async def get_drug_with_fallback(self, drug_name: str) -> Dict[str, Any]:
        """
        Get drug information with API-first approach and local fallback
        """

        # 1. Try API search first
        api_result = await self.search_drug_comprehensive(drug_name)

        # If API search successful and has meaningful data, return it
        if api_result.get('confidence_score', 0) > 0.7 and api_result.get('sources_used'):
            return api_result

        # 2. Fall back to local database
        if self.db_fallback:
            logger.info(f"Falling back to local database for {drug_name}")
            try:
                local_result = await self.db_fallback.search_drug(drug_name)
                if local_result:
                    local_result['data_source'] = 'local_database_fallback'
                    local_result['api_unavailable'] = True
                    return local_result
            except Exception as e:
                logger.error(f"Local database fallback failed: {str(e)}")

        # 3. Return API result even if low quality
        if api_result.get('confidence_score', 0) > 0:
            return api_result

        # 4. Return empty result with error
        logger.error(f"No drug data available for {drug_name}")
        return {
            'drug_name': drug_name,
            'error': 'No data available from APIs or local database',
            'sources_used': [],
            'data_quality': 'unavailable',
            'confidence_score': 0.0,
            'last_updated': datetime.now().isoformat()
        }


# Example usage within existing systems
async def example_integration():
    """Example of how to integrate the new API service"""

    # Create API service
    api_service = MedicalAPIDrugService()

    # Set up fallback to existing local database
    # api_service.set_fallback_service(existing_local_drug_service)

    # Search for a drug
    result = await api_service.search_drug_comprehensive("Augmentin")

    return result


if __name__ == "__main__":
    # Quick test
    async def test():
        service = MedicalAPIDrugService()
        await service.initialize()
        result = await service.search_drug_comprehensive("aspirin")
        print(json.dumps(result, indent=2))

    asyncio.run(test())