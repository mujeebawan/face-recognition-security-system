"""
Custom exceptions for NRTC Face AI library.
"""


class LicenseError(Exception):
    """Raised when license validation fails"""
    pass


class HardwareBindingError(Exception):
    """Raised when hardware binding validation fails"""
    pass


class LicenseExpiredError(LicenseError):
    """Raised when license has expired"""
    pass


class InvalidLicenseError(LicenseError):
    """Raised when license is invalid or tampered"""
    pass
