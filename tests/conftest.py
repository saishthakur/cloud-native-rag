import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cloudrag.config import Settings  # noqa: E402
from cloudrag.core.models import AccessPolicy, DocumentIn  # noqa: E402
from cloudrag.runtime import LocalPlatform  # noqa: E402


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=str(tmp_path), rate_limit_rps=0.0)


@pytest.fixture
def platform(settings):
    p = LocalPlatform(settings)
    yield p
    p.close()


DOCS = {
    "acme": [
        DocumentIn(doc_id="vendor-northwind", title="Vendor profile", source="erp://vendors",
                   text="Northwind Components is an approved supplier. Northwind Components is headquartered in Pune."),
        DocumentIn(doc_id="product-zx200", title="Product sheet", source="plm://products",
                   text="The ZX-200 Sensor is supplied by Northwind Components. "
                        "The ZX-200 Sensor carries a warranty period of 18 months."),
        DocumentIn(doc_id="project-aurora", title="Project charter", source="pmo://charters",
                   text="Project Aurora is led by Asha Kulkarni. Project Aurora uses the ZX-200 Sensor for "
                        "condition monitoring.\n\n[[acl roles=finance]]The approved budget for Project Aurora is "
                        "42.5 million rupees.[[/acl]]"),
        DocumentIn(doc_id="hr-asha", title="Employee record", source="hris://people",
                   text="Asha Kulkarni works in Research. Asha Kulkarni can be reached at asha.k@acme.example.com "
                        "or on +91 9845 123456."),
        DocumentIn(doc_id="exec-memo", title="Executive memo", source="docs://exec",
                   acl=AccessPolicy(roles=["exec"]),
                   text="The board approved the acquisition of Bluepeak Labs for an undisclosed sum."),
    ],
    "globex": [
        DocumentIn(doc_id="product-qk410", title="Product sheet", source="plm://products",
                   text="The QK-410 Drive is supplied by Helix Systems. The QK-410 Drive carries a warranty period "
                        "of 36 months."),
    ],
}


@pytest.fixture
def loaded(platform):
    for tenant, docs in DOCS.items():
        ctx = platform.context(tenant, "loader", ["ingest"])
        for d in docs:
            platform.gateway.ingest(ctx, d)
    assert platform.wait_indexed()
    return platform
