#ifndef ARPAREADER_HH
#define ARPAREADER_HH

#include <cstddef>
#include <string>
#include <vector>

#include "misc/SymbolMap.hh"


namespace fsalm {

/** Reads ARPA language model file.
 *
 * The following events are treated as error:
 * - An n-gram contains a symbol that is not introduced in 1-grams
 *
 * The following events are accepted silently:
 * - An n-gram has non-zero backoff, but no continuation
 * - Duplicate n-grams
 *
 */
class ArpaReader {
public:
    /** Construct a reader associated with a possible file. */
    ArpaReader(FILE *file = NULL);

    /** Reset the structure and prepare reading from file. */
    void reset(FILE *file);

    /** Read the header of the ARPA file. */
    void read_header();

    /** Read next ngram from the ARPA file.
     *
     * \return false if no more ngrams on this order.  Next call will
     * start reading next order.  See \ref end_reached to check end of
     * file.
     */
    bool read_ngram();

    /** Read all ngrams of the next order.
     * \return false if could not read more ngrams
     */
    bool read_order_ngrams(bool sort = false);

    /** Return the most recently read line from input. */
    std::string recent_line() const {
        return m_recent_line;
    }

    /** Number of ngrams ignored while reading. */
    int num_ignored() const {
        return m_num_ignored;
    }

    /** Configurable options. */
    struct Options {
        /** Throw an exception if header info does not match the
         * ngrams instead of just warning (default false). */
        bool throw_header_mismatch;

        /** Print progress messages (default true). */
        bool show_progress;

        /** Symbols to ignore. */
        std::vector<std::string> ignore_symbols;
    } opt;

    /** Information in the ARPA header. */
    struct Header {
        int order; //!< Order of the model
        int num_ngrams_total; //!< Total number of ngrams in the model

        /** The number of ngrams in each order (0 = 1-gram) */
        std::vector<int> num_ngrams;
    } header;

    /** The ngram read at the last call of read_ngram() */
    struct Ngram {
        std::vector<int> symbols; //!< The symbols of the ngram
        float log_prob; //!< The log-probability (base 10) of the ngram
        float backoff; //!< The backoff weight
        bool operator<(const Ngram &ngram) const
        {
            size_t i = 0;
            while (1) {
                if (i == symbols.size() && i == ngram.symbols.size())
                    return false;
                if (i == symbols.size())
                    return true;
                if (i == ngram.symbols.size())
                    return false;
                if (symbols[i] < ngram.symbols[i])
                    return true;
                if (ngram.symbols[i] < symbols[i])
                    return false;
                i++;
            }
        }

        /** The order of the ngram (1 = unigram, etc.) */
        int len() {
            return symbols.size();
        }
    } ngram;

    std::vector<Ngram> order_ngrams; //!< Ngrams read with read_order_ngrams()
    std::vector<int> sorted_order; //!< Indices of sorted ngrams

    misc::SymbolMap<std::string,int> *symbol_map; //!< The symbols in the model
    bool end_reached; //!< Have we reached the \end\ keyword

private:
    FILE *m_file; //!< The file to read from
    int m_current_order; //!< The order ngram just read from the file
    int m_ngrams_read; //!< The number of ngrams read for current order
    std::string m_recent_line; //!< The most recently read line from input
    int m_num_ignored; //!< Number of ngrams ignored while reading
};
};

#endif /* ARPAREADER_HH */
