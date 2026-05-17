# Operational Dst forecasting

# Setup

## Installing UV

See [UV documentation](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_1) for more details.

**Windows**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux/macOS**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Cloning the Repository

```bash
git clone https://github.com/ionutcatalinsandu/operational-dst-forecasting.git
cd operational-dst-forecasting
```

## Installing Dependencies

```bash
uv sync
```

## Activating the Environment

```bash
uv activate
```
