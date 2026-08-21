# Frontend

React-based user interface for Chennai FloodSense AI.

## Responsibilities

- Locality selection
- Interactive Chennai map
- Current flood-risk display
- Next-24-hour risk display
- Low / Medium / High risk visualization
- Early-warning presentation
- API error/loading states

## Backend integration

The frontend should consume the backend under `/api/v1`, primarily:

```text
GET /api/v1/localities
GET /api/v1/flood-risk/{locality}
```

Do not import Python ML code directly into the frontend. The backend is the application boundary.

## Ownership

Frontend developers should add the React application and map components here. ML implementation belongs only to the `machine-learning` branch.
