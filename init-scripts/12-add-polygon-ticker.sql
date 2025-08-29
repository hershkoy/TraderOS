-- Add polygon_ticker column to option_contracts table
-- This migration adds the polygon_ticker column for storing Polygon's ticker for API calls

-- Add polygon_ticker column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'option_contracts' 
        AND column_name = 'polygon_ticker'
    ) THEN
        ALTER TABLE option_contracts ADD COLUMN polygon_ticker TEXT;
        RAISE NOTICE 'Added polygon_ticker column to option_contracts table';
    ELSE
        RAISE NOTICE 'polygon_ticker column already exists in option_contracts table';
    END IF;
END $$;

-- Add comment for documentation
COMMENT ON COLUMN option_contracts.polygon_ticker IS 'Polygon ticker for API calls (e.g., O:QQQ250920C00300000)';
